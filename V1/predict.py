import sys
sys.path.append('../')
import torch
from fastNLP.core.predictor import Predictor

from paths import *
from load_data import *
from utils import norm_static_embedding
from V1.add_lattice import equip_chinese_ner_with_lexicon
from V1.parameters import args

if args.device!='cpu':
    assert args.device.isdigit()
    device = torch.device('cuda:{}'.format(args.device))
else:
    device = torch.device('cpu')
refresh_data = False
 
datasets,vocabs,embeddings = load_data_for_predict(
    f'/data1/nzw/CNER/{args.dataset}_conll',
    yangjie_rich_pretrain_unigram_path,
    yangjie_rich_pretrain_bigram_path,
    index_token=False,
    char_min_freq=args.char_min_freq,
    bigram_min_freq=args.bigram_min_freq,
    only_train_min_freq=args.only_train_min_freq
)

w_list = load_word_list(
    yangjie_rich_pretrain_word_path, _refresh=refresh_data,
    _cache_fp='cache/{}'.format(args.lexicon_name))

datasets,vocabs,embeddings = equip_chinese_ner_with_lexicon(
    datasets,vocabs,embeddings,
    w_list,yangjie_rich_pretrain_word_path,
    _refresh=refresh_data,_cache_fp=None,
    only_lexicon_in_train=args.only_lexicon_in_train,
    word_char_mix_embedding_path=char_and_word_path,
    number_normalized=args.number_normalized,
    lattice_min_freq=args.lattice_min_freq,
    only_train_min_freq=args.only_train_min_freq
)

# 设定field的输入输出性质和pad_val，即填充值
for v in datasets.values():
    if args.lattice:
        v.set_input('lattice','bigrams','seq_len','target')
        v.set_input('lex_num','pos_s','pos_e')
        v.set_target('target','seq_len','raw_chars')
        v.set_pad_val('lattice',vocabs['lattice'].padding_idx)
    else:
        v.set_input('chars','bigrams','seq_len','target')
        v.set_target('target','seq_len','raw_chars')

if args.norm_embed>0:
    for k,v in embeddings.items():
        norm_static_embedding(v,args.norm_embed)
if args.norm_lattice_embed>0:
    norm_static_embedding(embeddings['lattice'],args.norm_lattice_embed)

model_path = f'/data1/nzw/model_saved/FLAT/{args.dataset}/{args.saved_name}'
model = torch.load(model_path)

predictor = Predictor(model, batch_size=10)   # 这里的model是加载权重之后的model
pred_label_list = predictor.predict(datasets['train'])['pred']  # 预测结果
pred_target = datasets['train']['target']
pred_raw_char = datasets['train']['raw_chars']
idx2word = vocabs['label'].idx2word

def write_predict_result(f, chars, pred, target=''):
    if target:
        for c, p, t in zip(chars, pred, target):
            f.write(' '.join([c,p,t])+'\n')
    else:
        for c, p in zip(chars, pred):
            f.write(' '.join([c,p])+'\n')
    f.write('\n')

out_path = f'/home/ningziwei/Research/ArchFLAT/{args.dataset}.txt'
f = open(out_path, 'w', encoding='utf8')
for pred, target, raw_char in zip(pred_label_list, pred_target, pred_raw_char):
    pred = pred[0]
    pred = [idx2word[e] for e in pred]
    target = [idx2word[e] for e in target]
    write_predict_result(f, raw_char, target, pred)
f.close()