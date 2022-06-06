import torch
import re
from itertools import chain
from collections import Counter
from fastNLP import MetricBase
from fastNLP.core.metrics import _compute_f_pre_rec

def label_seq_to_ents(label_sequence, raw_chars, scheme="BIO"):
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
                    "start_idx": i,
                    "end_idx": j+1,
                    "text": ''.join(raw_chars[i:(j+1)]),
                    "label": tags[i][2:]
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
                            "start_idx": i,
                            "end_idx": j+1,
                            "text": ''.join(raw_chars[i:(j+1)]),
                            "label": tags[i][2:]
                        })
                        break
                    else:
                        j += 1
                i = j + 1
            elif tags[i].startswith("S"):
                entityMentions.append({
                    "start_idx": i,
                    "end_idx": i+1,
                    "text": ''.join(raw_chars[i:(i+1)]),
                    "label": tags[i][2:]
                })
                i += 1
            else:
                i += 1
    return entityMentions

def count_parallel(res):
    '''
    分割并列，统计有多少个并列实体
    '''
    def de_conj(res):
        '''分割连词'''
        conj = re.findall('和*或*及*与*[以及]*',res)
        conj = [c for c in conj if len(c)>0]
        res_ = []
        for c in conj:
            pos = res.find(c)
            res_.append(res[:pos])
            res = res[pos+1:]
        res_.append(res)
        return res_
    if '、' in res:
        res = res.split('、')
        end_res = de_conj(res[-1])
        return len(res) + len(end_res) - 1
    else:
        return len(de_conj(res))

def bound_compare(pre, gold):
    flag1 = pre['label'] == gold['label']
    flag2 = abs(pre['start_idx'] - gold['start_idx']) + abs(pre['end_idx'] - gold['end_idx'])
    return flag1 and flag2

def pre_contain_post(r1, r2):
    flag1 = r1['label'] == r2['label']
    flag2 = r1['start_idx']<=r2['start_idx'] and r1['end_idx']>=r2['end_idx']
    return flag1 and flag2

def _compute_f_pre_rec(tp, fn, fp, beta_square=1):
    r"""
    :param tp: int, true positive
    :param fn: int, false negative
    :param fp: int, false positive
    :return: (f, pre, rec)
    """
    pre = tp / (fp + tp + 1e-13)
    rec = tp / (fn + tp + 1e-13)
    f = (1 + beta_square) * pre * rec / (beta_square * pre + rec + 1e-13)
    return f, pre, rec

class ArchMetrics(MetricBase):
    '''
    针对建筑领域文本实体抽取和元素抽取的评价类
    '''
    def __init__(self, tag_vocab, enc_type='bio'):
        super(ArchMetrics, self).__init__()
        self.tag_vocab = tag_vocab
        self.enc_type = enc_type
        self.fp = 0
        self.tp = 0
        self.fn = 0

    def evaluate(self, pred, target, seq_len, raw_chars=None):
        r"""evaluate函数将针对一个批次的预测结果做评价指标的累计
        :param pred: [batch, seq_len] 或者 [batch, seq_len, len(tag_vocab)], 预测的结果
        :param target: [batch, seq_len], 真实值
        :param seq_len: [batch] 文本长度标记
        :return:
        """
        if len(pred.size()) == len(target.size()) + 1 and len(target.size()) == 2:
            num_classes = pred.size(-1)
            pred = pred.argmax(dim=-1)
            if (target >= num_classes).any():
                raise ValueError("A gold label passed to SpanBasedF1Metric contains an "
                                 "id >= {}, the number of classes.".format(num_classes))

        pred = pred.tolist()
        target = target.tolist()
        true_count, predict_count, gold_count = 0, 0, 0
        for i, (pre, gold, text) in enumerate(zip(pred, target, raw_chars)):
            # print('met 139', self.tag_vocab)
            # print(pre, gold, text)
            pre = pre[:int(seq_len[i])]
            gold = gold[:int(seq_len[i])]
            pre = [self.tag_vocab.to_word(tag) for tag in pre]
            gold = [self.tag_vocab.to_word(tag) for tag in gold]
            pre_res = label_seq_to_ents(pre, text, scheme=self.enc_type)
            gold_res = label_seq_to_ents(gold, text, scheme=self.enc_type)
            p, g = 0, 0
            pre_len = len(pre_res)
            gold_len = len(gold_res)
            while p<pre_len and g<gold_len:
                pr = pre_res[p]
                gol = gold_res[g]
                if pr == gol or bound_compare(pr, gol) < 2:
                    # 标注和预测相等
                    c = count_parallel(gold_res[g]['text'])
                    p += 1
                    g += 1
                    true_count += c
                    predict_count += c
                    gold_count += c
                elif pre_contain_post(pr, gol):
                    # 预测覆盖标注
                    while p<pre_len and g<gold_len and pre_contain_post(pre_res[p], gold_res[g]):
                        c = count_parallel(gold_res[g]['text'])
                        g += 1
                        predict_count += c
                        true_count += c
                        gold_count += c
                    p += 1
                elif pre_contain_post(gol, pr):
                    # 标注覆盖预测
                    while p<pre_len and g<gold_len and pre_contain_post(gold_res[g], pre_res[p]):
                        c = count_parallel(pre_res[p]['text'])
                        p += 1
                        gold_count += c
                        true_count += c
                        predict_count += c
                    g += 1
                elif pr['end_idx']<=gol['start_idx']:
                    p += 1
                    predict_count += 1
                elif pr['start_idx']>=gol['end_idx']:
                    g += 1
                    gold_count += 1
                else:
                    p += 1
                    g += 1
                    predict_count += 1
                    gold_count += 1
            predict_count += (pre_len - p)
            gold_count += (gold_len - g)
        
        self.tp += true_count
        self.fp += predict_count - true_count
        self.fn += gold_count - true_count
        # precision = true_count / max(predict_count, 1)
        # recall = true_count / max(gold_count, 1)
        # f1 = 2 * precision * recall / max((precision + recall), 1e-10)

    def get_metric(self, reset=True):
        '''所有的batch评价完成后计算整体指标'''
        f, pre, rec = _compute_f_pre_rec(
            self.tp, self.fn, self.fp)
        res = {
            'f1': round(f, 4)*100,
            'rec': round(rec, 4)*100,
            'pre': round(pre, 4)*100}
        if reset:
            self.fp = 0
            self.tp = 0
            self.fn = 0
        return res

