from fastNLP import cache_results


@cache_results(_cache_fp='need_to_defined_fp',_refresh=True)
def equip_chinese_ner_with_lexicon(datasets,vocabs,embeddings,w_list,word_embedding_path=None,
                                   only_lexicon_in_train=False,word_char_mix_embedding_path=None,
                                   number_normalized=False,
                                   lattice_min_freq=1,only_train_min_freq=0):
    '''
    datasets: {train:data_bundle(instance({chars ['科','技','全'...], target ['O','O'...], bigrams ['科技','技全'...], seq_lens 26})), test:, dev:}
    vocabs: {'char':Vovabulary, 'label':Vovabulary, 'bigram':Vovabulary}, 字符和id相互转换，idx2word, word2idx
    embeddings: char和bigram的预训练向量，embedding.embedding.state_dict()['weight']
    w_list: 词列表，['科技','智能',...]
    return datasets {
        'train': {chars, raw_chars, lexicons, lex_num, lex_s, lex_e, lattice}
        'dev':
        'test':
    }
    vocbs{
        'char': {'c1': id1, 'c2':id2}
        'bigram': Vocabulary
        'word': Vocabulary
        'lattice': Vocabulary
    }
    embeddings {
        'char':
        'bigram':
        'word':
        'lattice': StaticEmbedding
    }
    '''
    from fastNLP.core import Vocabulary
    from V0.utils_ import Trie, get_skip_path
    from functools import partial
    # from fastNLP.embeddings import StaticEmbedding
    from fastNLP_module import StaticEmbedding

    def normalize_char(inp):
        '''数字全部换成'0'，其他保持不变'''
        result = []
        for c in inp:
            if c.isdigit():
                result.append('0')
            else:
                result.append(c)

        return result
    def normalize_bigram(inp):
        '''数字全部换成'0'，其他保持不变'''
        result = []
        for bi in inp:
            if bi[0].isdigit(): bi[0] = '0'
            if bi[1].isdigit(): bi[1] = '0'
            result.append(bi)
        return result
    
    # 单词查找树，前缀树
    w_trie = Trie()
    for w in w_list: w_trie.insert(w)
    # 只保留外部词表中出现在训练集中的词，提高数据处理速度
    if only_lexicon_in_train:
        print('只加载在trian中出现过的词汇')
        lexicon_in_train = set()
        for s in datasets['train']['chars']:
            lexicon_in_s = w_trie.get_lexicon(s)
            for _,_,lexicon in lexicon_in_s:
                lexicon_in_train.add(''.join(lexicon))
        print('lexicon in train:{}'.format(len(lexicon_in_train)))
        print('i.e.: {}'.format(list(lexicon_in_train)[:10]))
        w_trie = Trie()
        for w in lexicon_in_train: w_trie.insert(w)

    import copy
    # 添加 lexicon 相关的 field
    for k,v in datasets.items():
        # print('74 add_lattice', v[0])
        # get_skip_path 得到句子中出现的所有词表中的词
        v.apply_field(partial(get_skip_path,w_trie=w_trie),'chars','lexicons')      # 给每个句子添加词汇信息
        # print('77 add_lattice', v[0])
        v.apply_field(copy.copy, 'chars','raw_chars')
        # print('79 add_lattice', v[0])
        v.add_seq_len('lexicons','lex_num')                                         # 统计词汇个数
        # print('81 add_lattice', v[0])
        v.apply_field(lambda x: list(map(lambda y: y[0], x)), 'lexicons', 'lex_s')  # 取所有词的开始位置
        # print('83 add_lattice', v[0])
        v.apply_field(lambda x: list(map(lambda y: y[1], x)), 'lexicons', 'lex_e')  # 取所有词的结束位置
        # print('85 add_lattice', v[0])
    # 构建 char 和 bigram 的文本-编号映射，标记其文本在训练数据、非训练数据的出现情况
    if number_normalized == 3:
        for k,v in datasets.items():    # datasets中的v是dataset类，dataset每行代表一个样本，每列表示一个特征
            v.apply_field(normalize_char,'chars','chars')   # 在原位置标准化chars的内容
        # 用于构建、存储和使用 str 到 int 的一一映射
        vocabs['char'] = Vocabulary()
        # 在train中但不在PLM词表中的词构建新的向量，微调时学习其表示；只在非训练集中的词直接对应[UNK]
        vocabs['char'].from_dataset(datasets['train'], field_name='chars',
                                no_create_entry_dataset=[datasets['dev'], datasets['test']])
        for k,v in datasets.items():
            v.apply_field(normalize_bigram,'bigrams','bigrams')
        vocabs['bigram'] = Vocabulary() 
        vocabs['bigram'].from_dataset(datasets['train'], field_name='bigrams',
                                  no_create_entry_dataset=[datasets['dev'], datasets['test']])
    if number_normalized == 1:
        for k,v in datasets.items():
            v.apply_field(normalize_char,'chars','chars')
        vocabs['char'] = Vocabulary()
        vocabs['char'].from_dataset(datasets['train'], field_name='chars',
                                no_create_entry_dataset=[datasets['dev'], datasets['test']])
    if number_normalized == 2:
        for k,v in datasets.items():
            v.apply_field(normalize_char,'chars','chars')
        vocabs['char'] = Vocabulary()
        vocabs['char'].from_dataset(datasets['train'], field_name='chars',
                                no_create_entry_dataset=[datasets['dev'], datasets['test']])

        for k,v in datasets.items():
            v.apply_field(normalize_bigram,'bigrams','bigrams')
        vocabs['bigram'] = Vocabulary()
        vocabs['bigram'].from_dataset(datasets['train'], field_name='bigrams',
                                  no_create_entry_dataset=[datasets['dev'], datasets['test']])


    def concat(ins):
        '''拼接chars和lexicons，得到作为输入的lattice'''
        chars = ins['chars']
        lexicons = ins['lexicons']
        result = chars + list(map(lambda x:x[2],lexicons))
        return result
    def get_pos_s(ins):
        '''拼接char和lex的开始位置'''
        lex_s = ins['lex_s']
        seq_len = ins['seq_len']
        pos_s = list(range(seq_len)) + lex_s
        return pos_s
    def get_pos_e(ins):
        '''拼接char和lex的结束位置'''
        lex_e = ins['lex_e']
        seq_len = ins['seq_len']
        pos_e = list(range(seq_len)) + lex_e
        return pos_e
    
    # print('139 add_lattice', datasets['train'][0])
    for k in datasets['train'][0].items(): print(k)
    # 添加 lattice 格式的文本、开始位置、结束位置
    for k,v in datasets.items():
        v.apply(concat, new_field_name='lattice')       # apply直接作用于每一个样本
        v.set_input('lattice')                          # 声明该field的数据是模型输入
        v.apply(get_pos_s, new_field_name='pos_s')
        v.apply(get_pos_e, new_field_name='pos_e')
        v.set_input('pos_s','pos_e')
    # print('148 add_lattice', datasets['train'][0])
    for k in datasets['train'][0].items(): print(k)


    # 构建 w_list 和 lattice 的文本-编号映射，datasets['train']['lattice']是列表，构建映射时会解析出里面的字符串
    word_vocab = Vocabulary()
    word_vocab.add_word_lst(w_list)
    vocabs['word'] = word_vocab

    lattice_vocab = Vocabulary()
    lattice_vocab.from_dataset(datasets['train'],field_name='lattice',
                               no_create_entry_dataset=[v for k,v in datasets.items() if k != 'train'])
    vocabs['lattice'] = lattice_vocab

    if word_embedding_path is not None:
        word_embedding = StaticEmbedding(word_vocab,word_embedding_path,word_dropout=0)
        embeddings['word'] = word_embedding

    if word_char_mix_embedding_path is not None:
        lattice_embedding = StaticEmbedding(lattice_vocab, word_char_mix_embedding_path,word_dropout=0.01,
                                            min_freq=lattice_min_freq,only_train_min_freq=only_train_min_freq)
        embeddings['lattice'] = lattice_embedding

    # 用对应的文本-编号映射将数据集中对应field内容替换为编号
    vocabs['char'].index_dataset(* (datasets.values()),
                             field_name='chars', new_field_name='chars')
    vocabs['bigram'].index_dataset(* (datasets.values()),
                               field_name='bigrams', new_field_name='bigrams')
    vocabs['label'].index_dataset(* (datasets.values()),
                              field_name='target', new_field_name='target')
    vocabs['lattice'].index_dataset(* (datasets.values()),
                                    field_name='lattice', new_field_name='lattice')

    return datasets,vocabs,embeddings
