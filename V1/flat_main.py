import os
import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.optim as optim
import collections
import fitlog

# from callbacks import FitlogCallback
from torch.optim.lr_scheduler import LambdaLR
from fastNLP import logger
from fastNLP import LRScheduler
from fastNLP import LossInForward
from fastNLP import FitlogCallback
from fastNLP.core import Trainer
from fastNLP.core import Callback
from fastNLP.core.metrics import AccuracyMetric
from fastNLP.core.callback import WarmupCallback,GradientClipCallback

# from ..paths import *
# from ..load_data import *
# from ..utils import get_peking_time, print_info
# from ..utils import norm_static_embedding
# from ..fastNLP_module import BertEmbedding
# from .add_lattice import equip_chinese_ner_with_lexicon
# from .models import Lattice_Transformer_SeqLabel
# from .metrics import ArchMetrics
# from .parameters import args

from paths import *
from load_data import *
from utils import get_peking_time, print_info
from utils import norm_static_embedding
from fastNLP_module import BertEmbedding
from V1.add_lattice import equip_chinese_ner_with_lexicon
from V1.models import Lattice_Transformer_SeqLabel
from V1.metrics import ArchMetrics
from V1.parameters import args

def main(args):
    use_fitlog = True
    load_dataset_seed = 100
    now_time = get_peking_time()
    if not use_fitlog:
        fitlog.debug()
    fitlog.set_log_dir('/home/ningziwei/Research/ArchFLAT/V1/logs')
    fitlog.add_hyper(load_dataset_seed,'load_dataset_seed')
    fitlog.set_rng_seed(load_dataset_seed)
    fitlog.set_log_dir('/home/ningziwei/Research/ArchFLAT/V1/logs')
    logger.add_file('/home/ningziwei/Research/ArchFLAT/V1/log/{}'.format(now_time),level='info')
    if args.test_batch == -1:
        args.test_batch = args.batch//2
    fitlog.add_hyper(now_time,'time')

    if args.device!='cpu':
        assert args.device.isdigit()
        device = torch.device('cuda:{}'.format(args.device))
    else:
        device = torch.device('cpu')
    refresh_data = False

    '''
    datasets: chars ['科','技','全'...], target ['O','O'...], bigrams ['科技','技全'...], seq_lens 26
    vocabs: {'char':Vovabulary, 'label':Vovabulary, 'bigram':Vovabulary}, idx2word, word2idx
    embeddings: chat和bigram的预训练向量，embedding.embedding.state_dict()['weight']
    '''
    args.epoch = 200
    cache_root = '/home/ningziwei/Research/ArchFLAT/V1/cache'
    cache_name = os.path.join(cache_root, args.dataset)
    datasets,vocabs,embeddings = load_data_for_train(
        f'/data1/nzw/CNER/{args.dataset}_conll',
        unigram_path, bigram_path,
        _refresh=refresh_data,index_token=False,
        _cache_fp=cache_name,
        char_min_freq=args.char_min_freq,
        bigram_min_freq=args.bigram_min_freq,
        only_train_min_freq=args.only_train_min_freq
    )

    if args.gaz_dropout < 0:
        args.gaz_dropout = args.embed_dropout

    args.hidden = args.head_dim * args.head
    args.ff = args.hidden * args.ff

    # print('用的词表的路径:{}'.format(word_path))
    # w_list: ['</s>','-unknown-','中国','记者','今天',...]
    w_list = load_word_list(
        word_path, _refresh=refresh_data,
        _cache_fp='cache/{}'.format(args.lexicon_name))

    cache_name = os.path.join(cache_root,args.dataset+'_embed')

    # for k in datasets['train'][0].items(): print(k)
    # 对实体抽取数据进行数据增强
    '''
        datasets { 'train': {chars, raw_chars, lexicons, lex_num, lex_s, lex_e, lattice},'dev':,'test':}
        vocbs{
            'char': {'c1': id1, 'c2':id2}
            'bigram': Vocabulary
            'word': Vocabulary
            'lattice': Vocabulary
        }
        embeddings {'char':,'bigram':,'word':,'lattice': StaticEmbedding}
    '''
    datasets,vocabs,embeddings = equip_chinese_ner_with_lexicon(
        datasets,vocabs,embeddings,
        w_list,word_path,
        _refresh=refresh_data,_cache_fp=cache_name,
        only_lexicon_in_train=args.only_lexicon_in_train,
        word_char_mix_embedding_path=char_and_word_path,
        number_normalized=args.number_normalized,
        lattice_min_freq=args.lattice_min_freq,
        only_train_min_freq=args.only_train_min_freq
    )

    # 设定field的输入输出性质和pad_val，即填充值
    for k, v in datasets.items():
        if args.lattice:
            v.set_input('lattice','bigrams','seq_len','target')
            v.set_input('lex_num','pos_s','pos_e')
            v.set_target('target','seq_len','raw_chars')
            v.set_pad_val('lattice',vocabs['lattice'].padding_idx)
        else:
            v.set_input('chars','bigrams','seq_len','target')
            v.set_target('target','seq_len','raw_chars')


    # print(embeddings['char'].embedding.weight[:10])
    if args.norm_embed>0:
        for k,v in embeddings.items():
            norm_static_embedding(v,args.norm_embed)
    if args.norm_lattice_embed>0:
        norm_static_embedding(embeddings['lattice'],args.norm_lattice_embed)

    mode = {}
    mode['debug'] = args.debug
    mode['gpumm'] = args.gpumm
    if args.debug or args.gpumm:
        fitlog.debug()
    dropout = collections.defaultdict(int)
    dropout['embed'] = args.embed_dropout
    dropout['gaz'] = args.gaz_dropout
    dropout['output'] = args.output_dropout
    dropout['pre'] = args.pre_dropout
    dropout['post'] = args.post_dropout
    dropout['ff'] = args.ff_dropout
    dropout['ff_2'] = args.ff_dropout_2
    dropout['attn'] = args.attn_dropout

    torch.backends.cudnn.benchmark = False
    fitlog.set_rng_seed(args.seed)
    torch.backends.cudnn.benchmark = False

    fitlog.add_hyper(args)

    if args.continue_train or args.status == 'test':
        model_path = f'/data1/nzw/model_saved/FLAT/{args.dataset}/{args.saved_name}'
        model = torch.load(model_path)
        bert_embedding = model.bert_embedding
    else:
        max_seq_len = max(*map(lambda x:max(x['seq_len']),datasets.values()))
        if args.model_type=='bert':
            model_dir = '/data1/nzw/model/cn-wwm'
        elif args.model_type=='bart':
            model_dir = '/data1/nzw/model/bart-base-chinese'
        bert_embedding = BertEmbedding(
            vocabs['lattice'], model_dir_or_name=model_dir,
            requires_grad=False, word_dropout=0.01,
            model_type=args.model_type
        )
        model = Lattice_Transformer_SeqLabel(
            embeddings['lattice'], embeddings['bigram'], 
            len(vocabs['label']), dropout, args,
            mode, device, vocabs, 
            max_seq_len=max_seq_len,
            bert_embedding=bert_embedding
        )

    # 参数初始化
    with torch.no_grad():
        print_info('{}init pram{}'.format('*'*15,'*'*15))
        for n,p in model.named_parameters():
            if 'bert' not in n and 'embedding' not in n and 'pos' not in n and 'pe' not in n \
                    and 'bias' not in n and 'crf' not in n and p.dim()>1:
                try:
                    if args.init == 'uniform':
                        nn.init.xavier_uniform_(p)
                        print_info('xavier uniform init:{}'.format(n))
                    elif args.init == 'norm':
                        print_info('xavier norm init:{}'.format(n))
                        nn.init.xavier_normal_(p)
                except:
                    print_info(n)
                    exit(1208)
        print_info('{}init pram{}'.format('*' * 15, '*' * 15))

    loss = LossInForward()
    arch_metric = ArchMetrics(vocabs['label'], args.encoding_type)
    metrics = [arch_metric]

    if args.self_supervised:
        chars_acc_metric = AccuracyMetric(
            pred='chars_pred',target='chars_target',seq_len='seq_len')
        chars_acc_metric.set_metric_name('chars_acc')
        metrics.append(chars_acc_metric)

    # 设置模型不同层的学习率
    bert_embedding_param = list(model.bert_embedding.parameters())
    bert_embedding_param_ids = list(map(id,bert_embedding_param))
    bigram_embedding_param = list(model.bigram_embed.parameters())
    gaz_embedding_param = list(model.lattice_embed.parameters())
    embedding_param = bigram_embedding_param
    if args.lattice:
        gaz_embedding_param = list(model.lattice_embed.parameters())
        embedding_param = embedding_param+gaz_embedding_param
    embedding_param_ids = list(map(id,embedding_param))
    non_embedding_param = list(filter(
        lambda x:id(x) not in embedding_param_ids and id(x) not in bert_embedding_param_ids,
                                        model.parameters()))
    param_ = [
        {'params': non_embedding_param}, 
        {'params': embedding_param, 'lr': args.lr*args.embed_lr_rate},
        {'params': bert_embedding_param,'lr':args.lr*args.bert_lr_rate}]

    # 设置优化函数
    if args.optim == 'adam':
        optimizer = optim.AdamW(param_,lr=args.lr,weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(param_,lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)

    fitlog_evaluate_dataset = {'test':datasets['test']}
    for ins in datasets['dev']:
        datasets['train'].append(ins)
    if args.test_train:
        fitlog_evaluate_dataset['train'] = datasets['train']
    evaluate_callback = FitlogCallback(fitlog_evaluate_dataset, verbose=1)
    lrschedule_callback = LRScheduler(lr_scheduler=LambdaLR(optimizer, lambda ep: 1 / (1 + 0.05*ep) ))
    clip_callback = GradientClipCallback(clip_type='value', clip_value=5)

    class Unfreeze_Callback(Callback):
        def __init__(self,bert_embedding,fix_epoch_num):
            super().__init__()
            self.bert_embedding = bert_embedding
            self.fix_epoch_num = fix_epoch_num
            assert self.bert_embedding.requires_grad == False

        def on_epoch_begin(self):
            if self.epoch == self.fix_epoch_num+1:
                self.bert_embedding.requires_grad = True

    callbacks = [
        evaluate_callback,
        lrschedule_callback,
        clip_callback
    ]
    if args.fix_bert_epoch != 0:
        callbacks.append(Unfreeze_Callback(bert_embedding,args.fix_bert_epoch))
    else:
        bert_embedding.requires_grad = True
    if args.warmup > 0 and args.model == 'transformer':
        callbacks.append(WarmupCallback(warmup=args.warmup))

    class record_best_test_callback(Callback):
        def __init__(self,trainer,result_dict):
            super().__init__()
            self.trainer222 = trainer
            self.result_dict = result_dict

        def on_valid_end(self, eval_result, metric_key, optimizer, better_result):
            print(eval_result['data_test']['SpanFPreRecMetric']['f'])

    # 训练模块
    if args.status == 'train':
        save_path = f'/data1/nzw/model_saved/FLAT/{args.dataset}'
        trainer = Trainer(
            datasets['train'],model,optimizer,loss,args.batch,
            n_epochs=args.epoch, dev_data=datasets['test'],
            metrics=metrics, save_path=save_path,
            device=device, callbacks=callbacks,
            dev_batch_size=args.test_batch,
            test_use_tqdm=False,check_code_level=-1,
            update_every=args.update_every)
        trainer.train()

    # 直接用训好的模型预测
    from fastNLP.core.predictor import Predictor

    def write_predict_result(f, chars, pred, target=''):
        if target:
            for c, p, t in zip(chars, pred, target):
                f.write(' '.join([c,p,t])+'\n')
        else:
            for c, p in zip(chars, pred):
                f.write(' '.join([c,p])+'\n')
        f.write('\n')

    if args.status == 'test':
        predictor = Predictor(model)   # 这里的model是加载权重之后的model

        test_label_list = predictor.predict(datasets['test'])['pred']  # 预测结果
        test_target = datasets['test']['target']
        test_raw_char = datasets['test']['raw_chars']
        # show_index = 0
        # print('643', vocabs['label'].idx2word)
        # print('643', vocabs['label'].word2idx)
        # print('645', test_label_list)
        # print('712', datasets['test'][:3]['target'][show_index])
        # print('645', test_label_list[show_index])
        # print(test_raw_char[0])
        # for ch in test_raw_char: print(ch)
        idx2word = vocabs['label'].idx2word
        out_path = f'/home/ningziwei/Research/ArchFLAT/{args.dataset}.txt'
        f = open(out_path, 'w', encoding='utf8')
        for raw_char, pred, target in zip(test_raw_char, test_label_list, test_target):
            print('316', raw_char)
            print('317', pred)
            print('318', target)
            pred = [idx2word[e] for e in pred[0]]
            target = [idx2word[e] for e in target]
            write_predict_result(f, raw_char, target, pred)
        f.close()

if __name__=='__main__':
    main(args)