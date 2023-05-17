from dataclasses import dataclass
from typing import List
import numpy as np
import pysrt
from pysrt import SubRipItem
from pysrt import SubRipTime
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from torch import Tensor


@dataclass
class SubItem:
    text: str
    start_time: int
    end_time: int


def to_timestamp(t: SubRipTime):
    return t.milliseconds + (t.seconds + (t.minutes + t.hours * 60) * 60) * 1000


def prepare_sub_item(item: SubRipItem):
    text = item.text
    text = text.replace('\n', ' ').replace('...', ' ')
    return SubItem(text, to_timestamp(item.start), to_timestamp(item.end))


def find_time_divergence(start1, end1, start2, end2):
    st1, et1 = sub_items1[start1].start_time, sub_items1[end1 - 1].end_time
    st2, et2 = sub_items2[start2].start_time, sub_items2[end2 - 1].end_time
    if et1 <= st2 or et2 <= st1:
        return 1
    total_dur = max(et1, et2) - min(st1, st2)
    if st1 <= st2 and et2 <= et1:
        return st2 - st1 + et1 - et2
    if st2 <= st1 and et1 <= et2:
        return st1 - st2 + et2 - et1
    return abs(st2 - st1) + abs(et2 - et1)


def construct_sentence(start, end, sub_items: List[SubItem]):
    sent = ''
    for i in range(start, end):
        sent = sent + ' ' + sub_items[i].text
    return sent


def find_semantic_divergence_metric(start1, end1, start2, end2):
    len1 = end1 - start1
    len2 = end2 - start2
    sent1 = construct_sentence(start1, end1, sub_items1)
    sent2 = construct_sentence(start2, end2, sub_items2)
    if not tensor_flag1[start1][len1]:
        tensor_dp1[start1][len1] = model.encode(sent1, convert_to_tensor=True)
        tensor_flag1[start1][len1] = True
    if not tensor_flag2[start2][len2]:
        tensor_dp2[start2][len2] = model.encode(sent2, convert_to_tensor=True)
        tensor_flag2[start2][len2] = True
    return (1 - util.cos_sim(tensor_dp1[start1][len1], tensor_dp2[start2][len2])[0][0]) / 2


def find_divergence_metric(start1, end1, start2, end2):
    time_div = find_time_divergence(start1, end1, start2, end2)
    semantic_div_metric = find_semantic_divergence_metric(start1, end1, start2, end2)
    # print(time_div * semantic_div_metric)
    # print(time_div * semantic_div_metric)
    return time_div * semantic_div_metric


def find_best_len(pos1, pos2, step):
    if step > 4:
        return 0, 0, 0.0
    best_divergence = 100000000
    best_l1, best_l2 = 0, 0
    for i in range(pos1, min(pos1 + 3, len(sub_items1)) + 1):
        for j in range(pos2, min(pos2 + 3, len(sub_items2)) + 1):
            l1 = i - pos1
            l2 = j - pos2
            if (l1 == 0 and l2 != 1) or (l1 != 1 and l2 == 0):
                continue
            val1 = find_divergence_metric(pos1, i, pos2, j)
            val2 = find_best_len(i, j, step + 1)[2]
            divergence_metric = val1 + val2
            if divergence_metric < best_divergence:
                best_divergence = divergence_metric
                best_l1, best_l2 = l1, l2
    return best_l1, best_l2, best_divergence


def sync():
    pos1, pos2 = 16, 26
    while pos1 < len(sub_items1) and pos2 < len(sub_items2):
        len1, len2, divergence = find_best_len(pos1, pos2, 0)
        sent1 = construct_sentence(pos1, pos1 + len1, sub_items1)
        sent2 = construct_sentence(pos2, pos2 + len2, sub_items2)
        print('-----------------------------------')
        print(sent1 + ' -------> ' + sent2)
        if len1 > 0:
            print('1: from {} to {}'.format(pos1, pos1 + len1 - 1))
        else:
            print('1: -')
        if len2 > 0:
            print('2: from {} to {}'.format(pos2, pos2 + len2 - 1))
        else:
            print('2: -')
        print('divergence = ', divergence)
        print('-----------------------------------')
        pos1 = pos1 + len1
        pos2 = pos2 + len2


model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
subs1 = pysrt.open('en.srt')
subs2 = pysrt.open('ru.srt')

# s1 = 'You will listen closely, and you will not judge me until I am finished. If you cannot commit to this, then please leave the room.'
# s2 = 'Вы будете слушать и не будете осуждать, и не станете делать какие-либо выводы, пока я не закончу. Если вы не можете такого себе позволить,'
# ten1 = model.encode(s1, convert_to_tensor=True)
# ten2 = model.encode(s2, convert_to_tensor=True)
# print(util.cos_sim(ten1, ten2))

sub_items1 = [prepare_sub_item(item) for item in subs1]
sub_items2 = [prepare_sub_item(item) for item in subs2]

tensor_dp1 = np.zeros((len(sub_items1), 5), dtype=Tensor)
tensor_dp2 = np.zeros((len(sub_items2), 5), dtype=Tensor)
tensor_flag1 = np.zeros((len(sub_items1), 5), dtype=bool)
tensor_flag2 = np.zeros((len(sub_items2), 5), dtype=bool)

# print(find_best_len(13, 21, 0))

sync()
# 375 + 167 / 4094
# 00:04:33,917        00:04:38,555
# 00:04:34,294        00:04:38,388
#
# 0.882708063820612332902112979733
