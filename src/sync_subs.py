import datetime
from dataclasses import dataclass
from typing import List
import sys
import numpy as np
import pysrt
import re
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


def drop_formatting(s):
    s = re.sub('<\\s*b\\s*>', '', s)
    s = re.sub('<\\s*/\\s*b\\s*>', '', s)
    s = re.sub('<\\s*i\\s*>', '', s)
    s = re.sub('<\\s*/\\s*i\\s*>', '', s)
    s = re.sub('<\\s*u\\s*>', '', s)
    s = re.sub('<\\s*/\\s*u\\s*>', '', s)
    s = re.sub('<\\s*font\\s*color\\s*=\\s*"[^"]*"\\s*>', '', s)
    s = re.sub('<\\s*/\\s*font\\s*>', '', s)
    return s


def to_timestamp(t: SubRipTime):
    return t.milliseconds + (t.seconds + (t.minutes + t.hours * 60) * 60) * 1000


def prepare_sub_item(item: SubRipItem):
    text = item.text
    text = text.replace('\n', ' ').replace('...', ' ')
    text = drop_formatting(text)
    return SubItem(text, to_timestamp(item.start), to_timestamp(item.end))


def construct_sentence(start, end, sub_items: List[SubItem]):
    sent = ''
    for i in range(start, end):
        sent = sent + ' ' + sub_items[i].text
    return sent.strip()


def calc_time_div(start1, end1, start2, end2):
    st1, et1 = sub_items1[start1].start_time, sub_items1[end1 - 1].end_time
    st2, et2 = sub_items2[start2].start_time, sub_items2[end2 - 1].end_time
    if et1 <= st2 or et2 <= st1:
        return 100000
    if st1 <= st2 and et2 <= et1:
        return st2 - st1 + et1 - et2
    if st2 <= st1 and et1 <= et2:
        return st1 - st2 + et2 - et1
    return abs(st2 - st1) + abs(et2 - et1)


def calc_semantic_div(start1, end1, start2, end2):
    len1 = end1 - start1
    len2 = end2 - start2
    sent1 = construct_sentence(start1, end1, sub_items1)
    sent2 = construct_sentence(start2, end2, sub_items2)
    if not tensor_calculated1[start1][len1]:
        tensors1[start1][len1] = model.encode(sent1, convert_to_tensor=True)
        tensor_calculated1[start1][len1] = True
    if not tensor_calculated2[start2][len2]:
        tensors2[start2][len2] = model.encode(sent2, convert_to_tensor=True)
        tensor_calculated2[start2][len2] = True
    return (1 - util.cos_sim(tensors1[start1][len1], tensors2[start2][len2])[0][0]) / 2


def calc_div(start1, end1, start2, end2):
    time_div = calc_time_div(start1, end1, start2, end2)
    semantic_div = calc_semantic_div(start1, end1, start2, end2)
    return time_div * semantic_div


def duration(start, end, sub_items: List[SubItem]):
    return sub_items[end - 1].end_time - sub_items[start].start_time


def fit(pos1, pos2):
    # finish cases
    if pos1 >= n1 and pos2 >= n2:
        return 0
    if pos1 >= n1:
        return finish_time - sub_items2[pos2].start_time
    if pos2 >= n2:
        return finish_time - sub_items1[pos1].start_time
    # dp check
    if dp_calculated[pos1][pos2]:
        return dp[pos1][pos2]
    # start time diff limit
    if sub_items1[pos1].start_time - sub_items2[pos2].start_time >= 5000:
        dp[pos1][pos2] = fit(pos1, pos2 + 1)
        restore[pos1][pos2] = 0, 1
        dp_calculated[pos1][pos2] = True
        return dp[pos1][pos2]
    if sub_items2[pos2].start_time - sub_items1[pos1].start_time >= 5000:
        dp[pos1][pos2] = fit(pos1 + 1, pos2)
        restore[pos1][pos2] = 1, 0
        dp_calculated[pos1][pos2] = True
        return dp[pos1][pos2]
    best_div, best_l1, best_l2 = 1000000000000, -1, -1
    for i in range(pos1, min(pos1 + 4, n1) + 1):
        for j in range(pos2, min(pos2 + 4, n2) + 1):
            l1 = i - pos1
            l2 = j - pos2
            # skip by one phrase at once
            if l1 == 0 and l2 != 1 or l2 == 0 and l1 != 1:
                continue
            # duration limit
            if duration(pos1, i, sub_items1) >= 10000 or duration(pos2, j, sub_items2) >= 10000:
                continue
            # main logic
            cur_div = calc_div(pos1, i, pos2, j) + fit(i, j)
            if cur_div < best_div:
                best_div = cur_div
                best_l1 = l1
                best_l2 = l2
    dp[pos1][pos2] = best_div
    restore[pos1][pos2] = best_l1, best_l2
    dp_calculated[pos1][pos2] = True
    return dp[pos1][pos2]


def format_time(t):
    return datetime.datetime.fromtimestamp(t / 1000, tz=datetime.timezone.utc).strftime("%H:%M:%S,%f")[:-3]


def construct_time_interval(pos1, l1, pos2, l2):
    if l1 != 0:
        start_time = sub_items1[pos1].start_time
        end_time = sub_items1[pos1 + l1 - 1].end_time
    else:
        start_time = sub_items2[pos2].start_time
        end_time = min(sub_items1[pos1].start_time, sub_items2[pos2 + l2 - 1].end_time)
    return format_time(start_time), format_time(end_time)


def sync():
    print(fit(0, 0))
    pos1, pos2 = 0, 0
    seq_number = 0
    with open(output_path, 'a') as out:
        while pos1 < n1 and pos2 < n2:
            l1, l2 = restore[pos1][pos2]
            seq_number += 1
            start_time, end_time = construct_time_interval(pos1, l1, pos2, l2)
            sentence1 = construct_sentence(pos1, pos1 + l1, sub_items1)
            sentence2 = construct_sentence(pos2, pos2 + l2, sub_items2)
            out.write(f"\n{seq_number}\n{start_time} --> {end_time}\n{sentence1}\n{sentence2}")
            pos1 = pos1 + l1
            pos2 = pos2 + l2


subs_path1 = sys.argv[1]
subs_path2 = sys.argv[2]
output_path = sys.argv[3]

model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
subs1 = pysrt.open(subs_path1)
subs2 = pysrt.open(subs_path2)

sub_items1 = [prepare_sub_item(item) for item in subs1]
sub_items2 = [prepare_sub_item(item) for item in subs2]
n1 = len(sub_items1)
n2 = len(sub_items2)
finish_time = sub_items1[-1].end_time

dp = np.zeros((n1, n2), dtype=float)
restore = np.zeros((n1, n2, 2), dtype=int)
dp_calculated = np.zeros((n1, n2), dtype=bool)

tensors1 = np.zeros((n1, 5), dtype=Tensor)
tensors2 = np.zeros((n2, 5), dtype=Tensor)
tensor_calculated1 = np.zeros((n1, 5), dtype=bool)
tensor_calculated2 = np.zeros((n2, 5), dtype=bool)

sys.setrecursionlimit(1000000)
sync()
