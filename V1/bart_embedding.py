
import torch
import numpy as np
from torch import nn
from itertools import chain
from fastNLP import Vocabulary
from fastNLP.modules import BertTokenizer
from .modeling_bart import BartModel

class _BartWordModel(nn.Module):
    def __init__(self, model_dir_or_name: str, vocab: Vocabulary, layers: str = '-1', pool_method: str = 'first',
                 include_cls_sep: bool = False, pooled_cls: bool = False, auto_truncate: bool = False, min_freq=2):
        super().__init__()

        if isinstance(layers, list):
            self.layers = [int(l) for l in layers]
        elif isinstance(layers, str):
            self.layers = list(map(int, layers.split(',')))
        else:
            raise TypeError("`layers` only supports str or list[int]")
        assert len(self.layers) > 0, "There is no layer selected!"

        neg_num_output_layer = -16384
        pos_num_output_layer = 0
        for layer in self.layers:
            if layer < 0:
                neg_num_output_layer = max(layer, neg_num_output_layer)
            else:
                pos_num_output_layer = max(layer, pos_num_output_layer)

        self.tokenizer = BertTokenizer.from_pretrained(model_dir_or_name)
        model = BartModel.from_pretrained(model_dir_or_name)
        self.encoder = model.encoder
        self._max_position_embeddings = model.config.max_position_embeddings
        # 检查encoder_layer_number是否合理
        # print(dir(model))
        # print(dir(model.encoder))
        # print(dir(model.decoder))
        # print('41 bart_emb', model.config.d_model)
        self.encoder.hidden_size = model.config.d_model
        encoder_layer_number = len(model.encoder.layers)
        for layer in self.layers:
            if layer < 0:
                assert -layer <= encoder_layer_number, f"The layer index:{layer} is out of scope for " \
                                                       f"a bert model with {encoder_layer_number} layers."
            else:
                assert layer <= encoder_layer_number, f"The layer index:{layer} is out of scope for " \
                                                     f"a bert model with {encoder_layer_number} layers."

        assert pool_method in ('avg', 'max', 'first', 'last')
        self.pool_method = pool_method
        self.include_cls_sep = include_cls_sep
        self.pooled_cls = pooled_cls
        self.auto_truncate = auto_truncate

        # 将所有vocab中word的wordpiece计算出来, 需要额外考虑[CLS]和[SEP]
        self._has_sep_in_vocab = '[SEP]' in vocab  # 用来判断传入的数据是否需要生成token_ids

        word_to_wordpieces = []
        word_pieces_lengths = []
        for word, index in vocab:
            if index == vocab.padding_idx:  # pad是个特殊的符号
                word = '[PAD]'
            elif index == vocab.unknown_idx:
                word = '[UNK]'
            elif vocab.word_count[word]<min_freq:
                word = '[UNK]'
            word_pieces = self.tokenizer.wordpiece_tokenizer.tokenize(word)
            word_pieces = self.tokenizer.convert_tokens_to_ids(word_pieces)
            word_to_wordpieces.append(word_pieces)
            word_pieces_lengths.append(len(word_pieces))
        self._cls_index = self.tokenizer.vocab['[CLS]']
        self._sep_index = self.tokenizer.vocab['[SEP]']
        self._word_pad_index = vocab.padding_idx
        self._wordpiece_pad_index = self.tokenizer.vocab['[PAD]']  # 需要用于生成word_piece
        self.word_to_wordpieces = np.array(word_to_wordpieces)
        self.register_buffer('word_pieces_lengths', torch.LongTensor(word_pieces_lengths))

    def forward(self, words):
        r"""
        :param words: torch.LongTensor, batch_size x max_len
        :return: num_layers x batch_size x max_len x hidden_size或者num_layers x batch_size x (max_len+2) x hidden_size
        """
        # print('bart 84', words.shape)
        with torch.no_grad():
            batch_size, max_word_len = words.size()
            word_mask = words.ne(self._word_pad_index)  # 为1的地方有word
            seq_len = word_mask.sum(dim=-1)
            batch_word_pieces_length = self.word_pieces_lengths[words].masked_fill(word_mask.eq(False), 0)  # batch_size x max_len
            word_pieces_lengths = batch_word_pieces_length.sum(dim=-1)  # batch_size
            max_word_piece_length = batch_word_pieces_length.sum(dim=-1).max().item()  # 表示word piece的长度(包括padding)
            if max_word_piece_length + 2 > self._max_position_embeddings:
                if self.auto_truncate:
                    word_pieces_lengths = word_pieces_lengths.masked_fill(
                        word_pieces_lengths + 2 > self._max_position_embeddings,
                        self._max_position_embeddings - 2)
                else:
                    raise RuntimeError(
                        "After split words into word pieces, the lengths of word pieces are longer than the "
                        f"maximum allowed sequence length:{self._max_position_embeddings} of bert. You can set "
                        f"`auto_truncate=True` for BertEmbedding to automatically truncate overlong input.")

            # +2是由于需要加入[CLS]与[SEP]
            word_pieces = words.new_full((batch_size, min(max_word_piece_length + 2, self._max_position_embeddings)),
                                         fill_value=self._wordpiece_pad_index)
            attn_masks = torch.zeros_like(word_pieces)
            # 1. 获取words的word_pieces的id，以及对应的span范围
            word_indexes = words.cpu().numpy()
            for i in range(batch_size):
                # print('bart 109', word_indexes[i, :seq_len[i]])
                word_pieces_i = list(chain(*self.word_to_wordpieces[word_indexes[i, :seq_len[i]]]))
                # print('bart 111', word_pieces_i)
                if self.auto_truncate and len(word_pieces_i) > self._max_position_embeddings - 2:
                    word_pieces_i = word_pieces_i[:self._max_position_embeddings - 2]
                word_pieces[i, 1:word_pieces_lengths[i] + 1] = torch.LongTensor(word_pieces_i)
                attn_masks[i, :word_pieces_lengths[i] + 2].fill_(1)
            # print('471 bert_embed', word_pieces.shape)
            # 添加[cls]和[sep]
            word_pieces[:, 0].fill_(self._cls_index)
            batch_indexes = torch.arange(batch_size).to(words)
            word_pieces[batch_indexes, word_pieces_lengths + 1] = self._sep_index
            if self._has_sep_in_vocab:  # 但[SEP]在vocab中出现应该才会需要token_ids
                sep_mask = word_pieces.eq(self._sep_index).long()  # batch_size x max_len
                print('479 bert_embed', sep_mask)
                sep_mask_cumsum = sep_mask.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
                print('481 bert_embed', sep_mask_cumsum)
                token_type_ids = sep_mask_cumsum.fmod(2)
                print('483 bert_embed', token_type_ids)
                token_type_ids = token_type_ids[:, :1].__xor__(token_type_ids)  # 如果开头是奇数，则需要flip一下结果，因为需要保证开头为0
                print('485 bert_embed', token_type_ids)
            else:
                token_type_ids = torch.zeros_like(word_pieces)
        # 2. 获取hidden的结果，根据word_pieces进行对应的pool计算
        # all_outputs: [batch_size x max_len x hidden_size, batch_size x max_len x hidden_size, ...]
        # print('490 bert_embed', token_type_ids.shape)
        '''
        bert_outputs: layer_num*batch_size*max_len*emb_size
        '''
        # bert_outputs, pooled_cls = self.encoder(word_pieces, token_type_ids=token_type_ids, attention_mask=attn_masks,
        #                                         output_all_encoded_layers=True)
        
        '''bart start'''
        dict = self.encoder(
            input_ids=word_pieces, attention_mask=attn_masks, 
            return_dict=True, output_hidden_states=True)
        bert_outputs = dict.hidden_states
        pooled_cls = bert_outputs[-1][:, 0]
        '''bart end'''

        if self.include_cls_sep:
            s_shift = 1
            outputs = bert_outputs[-1].new_zeros(len(self.layers), batch_size, max_word_len + 2,
                                                     bert_outputs[-1].size(-1))
        else:
            s_shift = 0
            outputs = bert_outputs[-1].new_zeros(len(self.layers), batch_size, max_word_len,
                                                 bert_outputs[-1].size(-1))
        batch_word_pieces_cum_length = batch_word_pieces_length.new_zeros(batch_size, max_word_len + 1)
        batch_word_pieces_cum_length[:, 1:] = batch_word_pieces_length.cumsum(dim=-1)  # batch_size x max_len

        if self.pool_method == 'first':
            batch_word_pieces_cum_length = batch_word_pieces_cum_length[:, :seq_len.max()]
            batch_word_pieces_cum_length.masked_fill_(batch_word_pieces_cum_length.ge(max_word_piece_length), 0)
            _batch_indexes = batch_indexes[:, None].expand((batch_size, batch_word_pieces_cum_length.size(1)))
        elif self.pool_method == 'last':
            batch_word_pieces_cum_length = batch_word_pieces_cum_length[:, 1:seq_len.max()+1] - 1
            batch_word_pieces_cum_length.masked_fill_(batch_word_pieces_cum_length.ge(max_word_piece_length), 0)
            _batch_indexes = batch_indexes[:, None].expand((batch_size, batch_word_pieces_cum_length.size(1)))

        for l_index, l in enumerate(self.layers):
            output_layer = bert_outputs[l]
            real_word_piece_length = output_layer.size(1) - 2
            if max_word_piece_length > real_word_piece_length:  # 如果实际上是截取出来的
                paddings = output_layer.new_zeros(batch_size,
                                                  max_word_piece_length - real_word_piece_length,
                                                  output_layer.size(2))
                output_layer = torch.cat((output_layer, paddings), dim=1).contiguous()
            # 从word_piece collapse到word的表示
            truncate_output_layer = output_layer[:, 1:-1]  # 删除[CLS]与[SEP] batch_size x len x hidden_size
            if self.pool_method == 'first':
                tmp = truncate_output_layer[_batch_indexes, batch_word_pieces_cum_length]
                tmp = tmp.masked_fill(word_mask[:, :batch_word_pieces_cum_length.size(1), None].eq(False), 0)
                outputs[l_index, :, s_shift:batch_word_pieces_cum_length.size(1)+s_shift] = tmp
            elif self.pool_method == 'last':
                tmp = truncate_output_layer[_batch_indexes, batch_word_pieces_cum_length]
                tmp = tmp.masked_fill(word_mask[:, :batch_word_pieces_cum_length.size(1), None].eq(False), 0)
                outputs[l_index, :, s_shift:batch_word_pieces_cum_length.size(1)+s_shift] = tmp
            elif self.pool_method == 'max':
                for i in range(batch_size):
                    for j in range(seq_len[i]):
                        start, end = batch_word_pieces_cum_length[i, j], batch_word_pieces_cum_length[i, j + 1]
                        outputs[l_index, i, j + s_shift], _ = torch.max(truncate_output_layer[i, start:end], dim=-2)
            else:
                for i in range(batch_size):
                    for j in range(seq_len[i]):
                        start, end = batch_word_pieces_cum_length[i, j], batch_word_pieces_cum_length[i, j + 1]
                        outputs[l_index, i, j + s_shift] = torch.mean(truncate_output_layer[i, start:end], dim=-2)
            if self.include_cls_sep:
                if l in (len(bert_outputs) - 1, -1) and self.pooled_cls:
                    outputs[l_index, :, 0] = pooled_cls
                else:
                    outputs[l_index, :, 0] = output_layer[:, 0]
                outputs[l_index, batch_indexes, seq_len + s_shift] = output_layer[batch_indexes, word_pieces_lengths + s_shift]
        # 3. 最终的embedding结果
        # print('bart 205', outputs.shape)
        return outputs

    def save(self, folder):
        """
        给定一个folder保存pytorch_model.bin, config.json, vocab.txt
        :param str folder:
        :return:
        """
        self.tokenizer.save_pretrained(folder)
        self.encoder.save_pretrained(folder)

# class FBartEncoder(Seq2SeqEncoder):
#     def __init__(self, encoder):
#         super().__init__()
#         assert isinstance(encoder, BartEncoder)
#         self.bart_encoder = encoder

#     def forward(self, src_tokens, mask):
#         # 第i行前 src_seq_len[i] 的mask是True，后面的是False，长度是max(src_seq_len)或max_len
#         dict = self.bart_encoder(input_ids=src_tokens, attention_mask=mask, return_dict=True,
#                                  output_hidden_states=True)
#         encoder_outputs = dict.last_hidden_state
#         hidden_states = dict.hidden_states
#         return encoder_outputs, hidden_states

        # mask = seq_len_to_mask(src_seq_len, max_len=src_tokens.size(1))
        # dict = self.bart_encoder(input_ids=src_tokens, attention_mask=mask, return_dict=True,
        #                          output_hidden_states=True)
        # encoder_outputs = dict.last_hidden_state
        # hidden_states = dict.hidden_states
        # return encoder_outputs, mask, hidden_states
