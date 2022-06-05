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
from fastNLP.core.metrics import SpanFPreRecMetric,AccuracyMetric
from fastNLP.core.callback import WarmupCallback,GradientClipCallback,EarlyStopCallback

from paths import *
from load_data import *
from utils import get_peking_time, print_info
from fastNLP_module import BertEmbedding
from V1.add_lattice import equip_chinese_ner_with_lexicon
from V1.models import BERT_SeqLabel
from V1.models import Lattice_Transformer_SeqLabel, Transformer_SeqLabel
from V1.parameters import args

use_fitlog = True
eval_begin_epoch = 18
if not use_fitlog:
    fitlog.debug()
fitlog.set_log_dir('logs')
load_dataset_seed = 100
fitlog.add_hyper(load_dataset_seed,'load_dataset_seed')
fitlog.set_rng_seed(load_dataset_seed)

# fitlog.commit(__file__,fit_msg='绝对位置用新的了')
fitlog.set_log_dir('logs')
now_time = get_peking_time()
logger.add_file('log/{}'.format(now_time),level='info')
if args.test_batch == -1:
    args.test_batch = args.batch//2
fitlog.add_hyper(now_time,'time')

if args.device!='cpu':
    assert args.device.isdigit()
    device = torch.device('cuda:{}'.format(args.device))
else:
    device = torch.device('cpu')
refresh_data = False
# for k,v in args.__dict__.items():
#     print_info('{}:{}'.format(k,v))

raw_dataset_cache_name = os.path.join('cache',args.dataset)

if args.dataset == 'msra':
    '''
    datasets: chars ['科','技','全'...], target ['O','O'...], bigrams ['科技','技全'...], seq_lens 26
    vocabs: {'char':Vovabulary, 'label':Vovabulary, 'bigram':Vovabulary}, idx2word, word2idx
    embeddings: chat和bigram的预训练向量，embedding.embedding.state_dict()['weight']
    '''
    datasets,vocabs,embeddings = load_msra_ner_1(
        msra_ner_cn_path,
        yangjie_rich_pretrain_unigram_path,
        yangjie_rich_pretrain_bigram_path,
        _refresh=refresh_data,index_token=False,
        _cache_fp=raw_dataset_cache_name,
        char_min_freq=args.char_min_freq,
        bigram_min_freq=args.bigram_min_freq,
        only_train_min_freq=args.only_train_min_freq
    )
elif args.dataset == 'code_verb':
    args.epoch = 200
    datasets,vocabs,embeddings = load_ontonotes4ner(
        '/data1/nzw/CNER/code_verb_conll',
        yangjie_rich_pretrain_unigram_path,
        yangjie_rich_pretrain_bigram_path,
        _refresh=refresh_data,index_token=False,
        _cache_fp=raw_dataset_cache_name,
        char_min_freq=args.char_min_freq,
        bigram_min_freq=args.bigram_min_freq,
        only_train_min_freq=args.only_train_min_freq
    )

# print('flat_main 256', datasets['train'])

if args.gaz_dropout < 0:
    args.gaz_dropout = args.embed_dropout

args.hidden = args.head_dim * args.head
args.ff = args.hidden * args.ff


if args.lexicon_name == 'lk':
    yangjie_rich_pretrain_word_path = lk_word_path_2

print('用的词表的路径:{}'.format(yangjie_rich_pretrain_word_path))
'''
w_list: ['</s>','-unknown-','中国','记者','今天',...]
'''
w_list = load_yangjie_rich_pretrain_word_list(yangjie_rich_pretrain_word_path,
                                              _refresh=refresh_data,
                                              _cache_fp='cache/{}'.format(args.lexicon_name))

cache_name = os.path.join('cache',(
        args.dataset+'_lattice'+'_only_train:{}'+
        '_trainClip:{}'+'_norm_num:{}'+'char_min_freq{}'+
        'bigram_min_freq{}'+'word_min_freq{}'+'only_train_min_freq{}'+
        'number_norm{}'+'lexicon_{}'+'load_dataset_seed_{}'
    ).format(args.only_lexicon_in_train,
        args.train_clip,args.number_normalized,args.char_min_freq,
        args.bigram_min_freq,args.word_min_freq,args.only_train_min_freq,
        args.number_normalized,args.lexicon_name,load_dataset_seed
    )
)

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
        w_list,yangjie_rich_pretrain_word_path,
        _refresh=refresh_data,_cache_fp=cache_name,
        only_lexicon_in_train=args.only_lexicon_in_train,
        word_char_mix_embedding_path=yangjie_rich_pretrain_char_and_word_path,
        number_normalized=args.number_normalized,
        lattice_min_freq=args.lattice_min_freq,
        only_train_min_freq=args.only_train_min_freq
    )

# print('299 train:{}'.format(len(datasets['train'])))
avg_seq_len = 0
avg_lex_num = 0
avg_seq_lex = 0
train_seq_lex = []
dev_seq_lex = []
test_seq_lex = []
train_seq = []
dev_seq = []
test_seq = []
# 统计句子长度
def count_sent_len():
    for k,v in datasets.items():
        max_seq_len = 0
        max_lex_num = 0
        max_seq_lex = 0
        max_seq_len_i = -1
        for i in range(len(v)):
            if max_seq_len < v[i]['seq_len']:
                max_seq_len = v[i]['seq_len']
                max_seq_len_i = i
            # max_seq_len = max(max_seq_len,v[i]['seq_len'])
            max_lex_num = max(max_lex_num,v[i]['lex_num'])
            max_seq_lex = max(max_seq_lex,v[i]['lex_num']+v[i]['seq_len'])

            avg_seq_len+=v[i]['seq_len']
            avg_lex_num+=v[i]['lex_num']
            avg_seq_lex+=(v[i]['seq_len']+v[i]['lex_num'])
            if k == 'train':
                train_seq_lex.append(v[i]['lex_num']+v[i]['seq_len'])
                train_seq.append(v[i]['seq_len'])
                if v[i]['seq_len'] >200:
                    pass
                    # print('train里这个句子char长度已经超了200了')
                    # print(''.join(list(map(lambda x:vocabs['char'].to_word(x),v[i]['chars']))))
                else:
                    if v[i]['seq_len']+v[i]['lex_num']>400:
                        pass
                        # print('train里这个句子char长度没超200，但是总长度超了400')
                        # print(''.join(list(map(lambda x: vocabs['char'].to_word(x), v[i]['chars']))))
            if k == 'dev':
                dev_seq_lex.append(v[i]['lex_num']+v[i]['seq_len'])
                dev_seq.append(v[i]['seq_len'])
            if k == 'test':
                test_seq_lex.append(v[i]['lex_num']+v[i]['seq_len'])
                test_seq.append(v[i]['seq_len'])

        print('{} 最长的句子是:{}'.format(k,list(map(lambda x:vocabs['char'].to_word(x),v[max_seq_len_i]['chars']))))
        print('{} max_seq_len:{}'.format(k, max_seq_len))
        print('{} max_lex_num:{}'.format(k, max_lex_num))
        print('{} max_seq_lex:{}'.format(k, max_seq_lex))
# count_sent_len()

max_seq_len = max(*map(lambda x:max(x['seq_len']),datasets.values()))

# 设定field的输入输出性质和pad_val，即填充值
for k, v in datasets.items():
    if args.lattice:
        v.set_input('lattice','bigrams','seq_len','target')
        v.set_input('lex_num','pos_s','pos_e')
        v.set_target('target','seq_len')
        v.set_pad_val('lattice',vocabs['lattice'].padding_idx)
    else:
        v.set_input('chars','bigrams','seq_len','target')
        v.set_target('target', 'seq_len')

from utils import norm_static_embedding
# print(embeddings['char'].embedding.weight[:10])
if args.norm_embed>0:
    print('381 embedding:{}'.format(embeddings['char'].embedding.weight.size()))
    print('norm embedding')
    for k,v in embeddings.items():
        norm_static_embedding(v,args.norm_embed)

if args.norm_lattice_embed>0:
    print('387 embedding:{}'.format(embeddings['lattice'].embedding.weight.size()))
    print('norm lattice embedding')
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

if args.continue_train:
    model_path = f'/data1/nzw/model_saved/FLAT/{args.dataset}/best_Lattice_Transformer_SeqLabel'
    model = torch.load(model_path)
    bert_embedding = model.bert_embedding
else:
    if args.model == 'transformer':
        if args.lattice:
            if args.use_bert:
                if args.model_type=='bert':
                    model_dir = '/data1/nzw/model/cn-wwm'
                elif args.model_type=='bart':
                    model_dir = '/data1/nzw/model/bart-base-chinese'
                bert_embedding = BertEmbedding(
                    vocabs['lattice'], model_dir_or_name=model_dir,
                    requires_grad=False, word_dropout=0.01,
                    model_type=args.model_type
                )
            else:
                bert_embedding = None
            if args.only_bert:
                model = BERT_SeqLabel(bert_embedding,len(vocabs['label']),vocabs,args.after_bert)
            else:
                model = Lattice_Transformer_SeqLabel(
                    embeddings['lattice'], embeddings['bigram'], 
                    args.hidden, len(vocabs['label']), args.head, args.layer, 
                    args.use_abs_pos, args.use_rel_pos, args.learn_pos, args.add_pos,
                    args.pre, args.post, args.ff, args.scaled, dropout, 
                    args.use_bigram, mode, device, vocabs, 
                    max_seq_len=max_seq_len, rel_pos_shared=args.rel_pos_shared,
                    k_proj=args.k_proj, q_proj=args.q_proj,
                    v_proj=args.v_proj, r_proj=args.r_proj,
                    self_supervised=args.self_supervised, attn_ff=args.attn_ff,
                    pos_norm=args.pos_norm, ff_activate=args.ff_activate,
                    abs_pos_fusion_func=args.abs_pos_fusion_func,
                    embed_dropout_pos=args.embed_dropout_pos,
                    four_pos_shared=args.four_pos_shared,
                    four_pos_fusion=args.four_pos_fusion,
                    four_pos_fusion_shared=args.four_pos_fusion_shared,
                    bert_embedding=bert_embedding
                )
        else:
            model = Transformer_SeqLabel(
                embeddings['lattice'], embeddings['bigram'], args.hidden, len(vocabs['label']),
                args.head, args.layer, args.use_abs_pos,args.use_rel_pos,
                args.learn_pos, args.add_pos,
                args.pre, args.post, args.ff, args.scaled, dropout, args.use_bigram,
                mode,device,vocabs, max_seq_len=max_seq_len,
                rel_pos_shared=args.rel_pos_shared,
                k_proj=args.k_proj, q_proj=args.q_proj,
                v_proj=args.v_proj, r_proj=args.r_proj,
                self_supervised=args.self_supervised,
                attn_ff=args.attn_ff, pos_norm=args.pos_norm,
                ff_activate=args.ff_activate,
                abs_pos_fusion_func=args.abs_pos_fusion_func,
                embed_dropout_pos=args.embed_dropout_pos
            )
        # print(Transformer_SeqLabel.encoder.)
    elif args.model =='lstm':
        model = LSTM_SeqLabel_True(embeddings['char'],embeddings['bigram'],embeddings['bigram'],args.hidden,
                                len(vocabs['label']),
                            bidirectional=True,device=device,
                            embed_dropout=args.embed_dropout,output_dropout=args.output_dropout,use_bigram=True,
                            debug=args.debug)

# print('flat_main 495 model', model)

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
encoding_type = 'bio'
if args.dataset == 'weibo': encoding_type = 'bio'
f1_metric = SpanFPreRecMetric(vocabs['label'],pred='pred',target='target',seq_len='seq_len',encoding_type=encoding_type)
acc_metric = AccuracyMetric(pred='pred',target='target',seq_len='seq_len',)
acc_metric.set_metric_name('label_acc')
metrics = [
    f1_metric,
    acc_metric]

if args.self_supervised:
    chars_acc_metric = AccuracyMetric(pred='chars_pred',target='chars_target',seq_len='seq_len')
    chars_acc_metric.set_metric_name('chars_acc')
    metrics.append(chars_acc_metric)

if args.see_param:
    for n,p in model.named_parameters():
        print_info('{}:{}'.format(n,p.size()))
    print_info('see_param mode: finish')
    if not args.debug: exit(1208)

if not args.only_bert:
    if not args.use_bert:
        bigram_embedding_param = list(model.bigram_embed.parameters())
        gaz_embedding_param = list(model.lattice_embed.parameters())
        embedding_param = bigram_embedding_param
        if args.lattice:
            gaz_embedding_param = list(model.lattice_embed.parameters())
            embedding_param = embedding_param+gaz_embedding_param
        embedding_param_ids = list(map(id,embedding_param))
        non_embedding_param = list(filter(lambda x:id(x) not in embedding_param_ids,model.parameters()))
        param_ = [{'params': non_embedding_param}, {'params': embedding_param, 'lr': args.lr * args.embed_lr_rate}]
    else:
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
        param_ = [{'params': non_embedding_param}, {'params': embedding_param, 'lr': args.lr * args.embed_lr_rate},
                  {'params':bert_embedding_param,'lr':args.bert_lr_rate*args.lr}]
else:
    non_embedding_param = model.parameters()
    embedding_param = []
    param_ = [{'params': non_embedding_param}, {'params': embedding_param, 'lr': args.lr * args.embed_lr_rate}]


if args.optim == 'adam':
    optimizer = optim.AdamW(param_,lr=args.lr,weight_decay=args.weight_decay)
elif args.optim == 'sgd':
    # optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,
    #                       weight_decay=args.weight_decay)
    optimizer = optim.SGD(param_,lr=args.lr,momentum=args.momentum,
                          weight_decay=args.weight_decay)

if args.dataset == 'msra':
    datasets['dev']  = datasets['test']
fitlog_evaluate_dataset = {'test':datasets['test']}
if args.test_train:
    fitlog_evaluate_dataset['train'] = datasets['train']
evaluate_callback = FitlogCallback(fitlog_evaluate_dataset,verbose=1)
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
if args.use_bert:
    if args.fix_bert_epoch != 0:
        callbacks.append(Unfreeze_Callback(bert_embedding,args.fix_bert_epoch))
    else:
        bert_embedding.requires_grad = True
# callbacks.append(EarlyStopCallback(args.early_stop))
if args.warmup > 0 and args.model == 'transformer':
    callbacks.append(WarmupCallback(warmup=args.warmup))

class record_best_test_callback(Callback):
    def __init__(self,trainer,result_dict):
        super().__init__()
        self.trainer222 = trainer
        self.result_dict = result_dict

    def on_valid_end(self, eval_result, metric_key, optimizer, better_result):
        print(eval_result['data_test']['SpanFPreRecMetric']['f'])


if args.status == 'train':
    save_path = f'/data1/nzw/model_saved/FLAT/{args.dataset}'
    trainer = Trainer(datasets['train'],model,optimizer,loss,args.batch,
                      n_epochs=args.epoch,
                      dev_data=datasets['dev'],
                      metrics=metrics,
                      save_path=save_path,
                      device=device,callbacks=callbacks,dev_batch_size=args.test_batch,
                      test_use_tqdm=False,check_code_level=-1,
                      update_every=args.update_every)
    trainer.train()

# 直接用训好的模型predict时的代码
from fastNLP.core.predictor import Predictor
def label_sequence_to_entities(label_sequence, texts, scheme="BIO"):
    '''
    word_sequence: ["w0", "w1", ...]
    label_sequence: list or torch.Tensor(sen_len) for BIOES or torch.Tensor(sen_len, 2) for SPAN
    scheme: "BIOES" or "SPAN"
    '''
    if type(label_sequence) is torch.Tensor:
        label_sequence = label_sequence.tolist()
    tags = label_sequence
    entityMentions = []
    count = len(label_sequence)
    i = 0
    if scheme.upper() == 'BIO':
        while i < count:
            if tags[i].startswith("B"):
                j = i + 1
                while j < count:
                    if tags[j].startswith("O") or tags[j].startswith("B"):
                        j -= 1
                        break
                    j += 1
                entityMentions.append({
                    "start_index": i,
                    "end_index": j+1,
                    "text": ' '.join(texts[i:(j+1)])
                })
                i = j + 1
            else:
                i += 1
    elif scheme.upper() == 'BIOES':
        while i < count:
            if tags[i].startswith("B"):
                j = i + 1
                while j < count:
                    if tags[j].startswith("O") or tags[j].startswith("B"):
                        j -= 1
                        break
                    elif tags[j].startswith("E"):
                        entityMentions.append({
                            "start_index": i,
                            "end_index": j+1,
                            "text": ''.join(texts[i:(j+1)])
                        })
                        break
                    else:
                        j += 1
                i = j + 1
            elif tags[i].startswith("S"):
                entityMentions.append({
                    "start_index": i,
                    "end_index": i+1,
                    "text": ''.join(texts[i:(i+1)])
                    # "label": tags[i][2:]
                })
                i += 1
            else:
                i += 1
    return entityMentions

def write_predict_result(f, chars, pred, target=''):
    if target:
        for c, p, t in zip(chars, pred, target):
            f.write(' '.join([c,p,t])+'\n')
    else:
        for c, p in zip(chars, pred):
            f.write(' '.join([c,p])+'\n')
    f.write('\n')

if args.status == 'test':
    model_path = f'/data1/nzw/model_saved/FLAT/{args.dataset}/{args.saved_name}'
    states = torch.load(model_path).state_dict()
    model.load_state_dict(states)
    predictor = Predictor(model)   # 这里的model是加载权重之后的model

    test_label_list = predictor.predict(datasets['test'])['pred']  # 预测结果
    test_target = datasets['test']['target']
    test_raw_char = datasets['test']['raw_chars']
    show_index = 0
    # print('643', vocabs['label'].idx2word)
    # print('643', vocabs['label'].word2idx)
    # print('645', test_label_list)
    # print('712', datasets['test'][:3]['target'][show_index])
    # print('645', test_label_list[show_index])
    # print(test_raw_char[0])
    # for ch in test_raw_char: print(ch)
    idx2word = vocabs['label'].idx2word
    out_path = f'/home/ningziwei/Research/FLAT/{args.dataset}.txt'
    f = open(out_path, 'w', encoding='utf8')
    for pred, target, raw_char in zip(test_label_list, test_target, test_raw_char):
        pred = pred[0]
        pred = [idx2word[e] for e in pred]
        target = [idx2word[e] for e in target]
        write_predict_result(f, raw_char, target, pred)
    f.close()



