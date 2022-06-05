import sys
sys.path.append('../')
from paths import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--clip_msra',action='store_true')

args = parser.parse_args()
yangjie_rich_pretrain_word_path = 'D:\Download\Dataset\FLAT/ctb.50d.vec'
yangjie_rich_pretrain_unigram_path = 'D:\Download\Dataset\FLAT/gigaword_chn.all.a2b.uni.ite50.vec'
yangjie_rich_pretrain_char_and_word_path = 'D:\Download\Dataset\FLAT/yangjie_word_char_mix.txt'

lexicon_f = open(yangjie_rich_pretrain_word_path,'r',encoding='utf8')
char_f = open(yangjie_rich_pretrain_unigram_path,'r',encoding='utf8')
output_f = open(yangjie_rich_pretrain_char_and_word_path,'w',encoding='utf8')

lexicon_lines = lexicon_f.readlines()
print('18', lexicon_lines[100])
print(len(lexicon_lines[100].strip().split()))
for l in lexicon_lines:
    l_split = l.strip().split()
    if len(l_split[0]) != 1:
        print(l.strip(),file=output_f)

char_lines = char_f.readlines()
for l in char_lines:
    print(l.strip(),file=output_f)

