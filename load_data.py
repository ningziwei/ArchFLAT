import os
from fastNLP import Vocabulary, DataSet
from fastNLP import cache_results
from fastNLP.io.loader import ConllLoader
from fastNLP_module import StaticEmbedding
from utils import get_bigrams

@cache_results(_cache_fp='cache/load_word_list',_refresh=False)
def load_word_list(embedding_path,drop_characters=True):
    f = open(embedding_path,'r')
    lines = f.readlines()
    w_list = []
    for line in lines:
        splited = line.strip().split(' ')
        w = splited[0]
        w_list.append(w)

    if drop_characters:
        w_list = list(filter(lambda x:len(x) != 1, w_list))

    return w_list

@cache_results(_cache_fp='need_to_defined_fp',_refresh=False)
def equip_chinese_ner_with_skip(datasets,vocabs,embeddings,w_list,word_embedding_path=None,
                                word_min_freq=1,only_train_min_freq=0):
    from utils_ import Trie,get_skip_path
    from functools import partial
    w_trie = Trie()
    for w in w_list:
        w_trie.insert(w)
    # for k,v in datasets.items():
    #     v.apply_field(partial(get_skip_path,w_trie=w_trie),'chars','skips')

    def skips2skips_l2r(chars,w_trie):
        '''
        :param lexicons: list[[int,int,str]]
        :return: skips_l2r
        '''
        # print(lexicons)
        # print('******')
        lexicons = get_skip_path(chars,w_trie=w_trie)
        # max_len = max(list(map(lambda x:max(x[:2]),lexicons)))+1 if len(lexicons) != 0 else 0
        result = [[] for _ in range(len(chars))]
        for lex in lexicons:
            s = lex[0]
            e = lex[1]
            w = lex[2]
            result[e].append([s,w])
        return result

    def skips2skips_r2l(chars,w_trie):
        '''
        :param lexicons: list[[int,int,str]]
        :return: skips_l2r
        '''
        lexicons = get_skip_path(chars,w_trie=w_trie)
        # max_len = max(list(map(lambda x:max(x[:2]),lexicons)))+1 if len(lexicons) != 0 else 0
        result = [[] for _ in range(len(chars))]
        for lex in lexicons:
            s = lex[0]
            e = lex[1]
            w = lex[2]
            result[s].append([e,w])
        return result

    for k,v in datasets.items():
        v.apply_field(partial(skips2skips_l2r,w_trie=w_trie),'chars','skips_l2r')

    for k,v in datasets.items():
        v.apply_field(partial(skips2skips_r2l,w_trie=w_trie),'chars','skips_r2l')

    # print(v['skips_l2r'][0])
    word_vocab = Vocabulary()
    word_vocab.add_word_lst(w_list)
    vocabs['word'] = word_vocab
    for k,v in datasets.items():
        v.apply_field(lambda x:[ list(map(lambda x:x[0],p)) for p in x],'skips_l2r','skips_l2r_source')
        v.apply_field(lambda x:[ list(map(lambda x:x[1],p)) for p in x], 'skips_l2r', 'skips_l2r_word')

    for k,v in datasets.items():
        v.apply_field(lambda x:[ list(map(lambda x:x[0],p)) for p in x],'skips_r2l','skips_r2l_source')
        v.apply_field(lambda x:[ list(map(lambda x:x[1],p)) for p in x], 'skips_r2l', 'skips_r2l_word')

    for k,v in datasets.items():
        v.apply_field(lambda x:list(map(len,x)), 'skips_l2r_word', 'lexicon_count')
        v.apply_field(lambda x:
                      list(map(lambda y:
                               list(map(lambda z:word_vocab.to_index(z),y)),x)),
                      'skips_l2r_word',new_field_name='skips_l2r_word')

        v.apply_field(lambda x:list(map(len,x)), 'skips_r2l_word', 'lexicon_count_back')

        v.apply_field(lambda x:
                      list(map(lambda y:
                               list(map(lambda z:word_vocab.to_index(z),y)),x)),
                      'skips_r2l_word',new_field_name='skips_r2l_word')

    if word_embedding_path is not None:
        word_embedding = StaticEmbedding(word_vocab,word_embedding_path,word_dropout=0)
        embeddings['word'] = word_embedding

    vocabs['char'].index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                             field_name='chars', new_field_name='chars')
    vocabs['bigram'].index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                               field_name='bigrams', new_field_name='bigrams')
    vocabs['label'].index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                              field_name='target', new_field_name='target')

    return datasets,vocabs,embeddings

# cache_results将其修饰用到的数据缓存到_cache_fp，提升重复调用时的速度
@cache_results(_cache_fp='cache/ontonotes4ner',_refresh=False)
def load_data_for_train(
    path,char_embedding_path=None,bigram_embedding_path=None,
    index_token=True,train_clip=False,char_min_freq=1,
    bigram_min_freq=1,only_train_min_freq=0
):
    train_path = os.path.join(path,'train.train')
    dev_path = os.path.join(path,'dev.dev')
    test_path = os.path.join(path,'test.test')

    loader = ConllLoader(['chars','target'])
    train_bundle = loader.load(train_path)
    dev_bundle = loader.load(dev_path)
    test_bundle = loader.load(test_path)

    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['dev'] = dev_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']

    datasets['train'].apply_field(get_bigrams,field_name='chars',new_field_name='bigrams')
    datasets['dev'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')

    datasets['train'].add_seq_len('chars')
    datasets['dev'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')

    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()
    
    char_vocab.from_dataset(datasets['train'],field_name='chars',
                            no_create_entry_dataset=[datasets['dev'],datasets['test']])
    bigram_vocab.from_dataset(datasets['train'],field_name='bigrams',
                              no_create_entry_dataset=[datasets['dev'],datasets['test']])
    label_vocab.from_dataset(datasets['train'],field_name='target')
    if index_token:
        char_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='chars',new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='bigrams',new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='target',new_field_name='target')

    vocabs = {}
    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab
    vocabs['bigram'] = bigram_vocab
    vocabs['label'] = label_vocab

    embeddings = {}
    if char_embedding_path is not None:
        char_embedding = StaticEmbedding(char_vocab,char_embedding_path,word_dropout=0.01,
                                         min_freq=char_min_freq,only_train_min_freq=only_train_min_freq)
        embeddings['char'] = char_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab,bigram_embedding_path,word_dropout=0.01,
                                           min_freq=bigram_min_freq,only_train_min_freq=only_train_min_freq)
        embeddings['bigram'] = bigram_embedding

    return datasets,vocabs,embeddings

def load_str_data(txt):
    '''预测过程中读取文本'''
    datasets = dict()
    datasets['train'] = DataSet({'chars':list(txt),'target':[['']*len(txt)]})
    datasets['train'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['train'].add_seq_len('chars')
    return datasets

def load_file_data(data_path):
    '''预测过程中读取文件'''
    loader = ConllLoader(['chars','target'])
    train_bundle = loader.load(data_path)

    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['train'].apply_field(get_bigrams,field_name='chars',new_field_name='bigrams')
    datasets['train'].add_seq_len('chars')

    return datasets


if __name__ == '__main__':
    pass