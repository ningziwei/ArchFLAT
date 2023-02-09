import os
import torch
from fastNLP.core.predictor import Predictor

from ..paths import *
from ..load_data import *
from .add_lattice import *
from .utils_ import Trie

def tag_ent(args):
    cache_root = '/home/ningziwei/Research/ArchFLAT/V1/cache'
    refresh_data = False

    args.data_path = '/data1/nzw/CNER/'
    w_list = load_word_list(word_path, _cache_fp=f'{cache_root}/{args.lexicon_name}')
    raw_dataset_cache_name = os.path.join(cache_root,args.dataset)
    datasets,vocabs,embeddings = load_data_for_train(
        f'/data1/nzw/CNER/{args.dataset}_conll',
        unigram_path, bigram_path,
        _refresh=refresh_data,index_token=False,
        _cache_fp=raw_dataset_cache_name,
        char_min_freq=args.char_min_freq,
        bigram_min_freq=args.bigram_min_freq,
        only_train_min_freq=args.only_train_min_freq
    )
    cache_name = os.path.join(cache_root,args.dataset+'_embed')
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
    idx2word = vocabs['label'].idx2word

    w_trie = Trie()
    for w in w_list: w_trie.insert(w)

    model_path = f'/data1/nzw/model_saved/FLAT/{args.dataset}/{args.saved_name}'
    model = torch.load(model_path)
    predictor = Predictor(model)

    def label_seq_to_dic(tags, raw_chars, scheme=args.encoding_type):
        '''
        raw_chars: ["w0", "w1", ...]
        tags: list or torch.Tensor(sen_len)
        scheme: "BIOES" or "BIO"
        '''
        if type(tags) is torch.Tensor:
            tags = tags.tolist()
        raw_chars = ''.join(raw_chars)
        res_dic = {'txt':raw_chars}
        seq_len = len(tags)
        def insert_cell(i,j):
            if tags[i][2:] in res_dic:
                res_dic[tags[i][2:]].append(
                    [i,j+1,raw_chars[i:(j+1)]])
            else:
                res_dic[tags[i][2:]] = [
                    [i,j+1,raw_chars[i:(j+1)]]]
        i = 0
        if scheme.upper() == 'BIO':
            while i < seq_len:
                if tags[i].startswith("B"):
                    j = i + 1
                    while j < seq_len:
                        if tags[j].startswith("O") or tags[j].startswith("B"):
                            j -= 1
                            break
                        j += 1
                    insert_cell(i,j)
                    i = j + 1
                else:
                    i += 1
        elif scheme.upper() == 'BIOES':
            while i < seq_len:
                if tags[i].startswith("B"):
                    j = i + 1
                    while j < seq_len:
                        if tags[j].startswith("O") or tags[j].startswith("B"):
                            j -= 1
                            break
                        elif tags[j].startswith("E"):
                            insert_cell(i,j)
                            break
                        else:
                            j += 1
                    i = j + 1
                elif tags[i].startswith("S"):
                    insert_cell(i,i)
                    i += 1
                else:
                    i += 1
        return res_dic

    def write_predict_result(f, chars, pred, target=''):
        if target:
            for c, p, t in zip(chars, pred, target):
                f.write(' '.join([c,p,t])+'\n')
        else:
            for c, p in zip(chars, pred):
                f.write(' '.join([c,p])+'\n')
        f.write('\n')

    def tag_data(input_, mode='txt', out_path=None):
        if mode == 'file':
            datasets = load_file_data(input_)
        else:
            datasets = load_str_data(input_)
        datasets = equip_pred_data_with_lexicon(datasets, w_trie, vocabs) 
        for _, v in datasets.items():
            if args.lattice:
                v.set_input('lattice','bigrams','seq_len','target')
                v.set_input('lex_num','pos_s','pos_e')
                v.set_target('target','seq_len','raw_chars')
                # 设定field的输入输出性质和pad_val，即填充值
                v.set_pad_val('lattice',vocabs['lattice'].padding_idx)
            else:
                v.set_input('chars','bigrams','seq_len','target')
                v.set_target('target','seq_len','raw_chars')

        # print('tag_ent 134', datasets['train'][0].fields)
        pred_result = predictor.predict(datasets['train'])['pred']  # 预测结果
        targets = datasets['train']['target']
        raw_chars = datasets['train']['raw_chars']

        if mode=='file' and out_path:
            f = open(out_path, 'w', encoding='utf8')
            for pred, target, raw_char in zip(pred_result, targets, raw_chars):
                pred = [idx2word[e] for e in pred[0]]
                target = [idx2word[e] for e in target]
                write_predict_result(f, raw_char, target, pred)
            f.close()
            return '预测完成'
        else:
            res = []
            for pred, raw_char in zip(pred_result, raw_chars):
                pred = [idx2word[e] for e in pred[0]]
                res.append(label_seq_to_dic(pred, raw_char))
            return res

    while True:
        input_ = input('输入文本或txt文件路径：')
        if input_.endswith('.txt'):
            out_path = input_.replace('.txt','.pred')
            res = tag_data(input_, 'file', out_path)
        else:
            res = tag_data(input_)
        print(res)
