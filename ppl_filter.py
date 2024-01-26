import json
import os
import re
j_list = []
filename = 'len_200_ppl.jsonl'
# filename = 'zhihu_ppl_bucket_ppl.jsonl'

from typing import List, Dict
import json

punc_pattern = r'[，。]'
# 使用这个模式创建一个正则表达式对象
punc_regex = re.compile(punc_pattern)
def is_miss_punctuation(text):
    max_length = 0
    for para in text.split("\n"):
        splits = punc_regex.split(para)
        # 找出分割后的最长文本的长度
        cur_max_length = max(len(s) for s in splits)
        max_length = max(max_length, cur_max_length)
        # print(f"{cur_max_length}: {splits}")
    return max_length > 100

def is_poem(text):
    split_text = re.split('。|，', text)
    len_list = list(map(lambda x: len(x.strip()), split_text))
    max_continue = 0
    start = 0
    count = 0
    # 连续短句出现 4 次且句子占比 > 0.5, 判定为诗词
    for i in range(len(len_list)):
        if 4 <= len_list[i] <= 7:
            count += 1
            continue
        else:
            max_continue = max(max_continue, i - start)
            start = i

    return max_continue >= 4 and count / len(len_list) > 0.5

def write_out(d, out_file):
    with open(out_file, 'w') as of:
        for x in d:
            j = json.dumps(x, ensure_ascii=False)
            of.write(f"{j}\n")

def write_out2(d: List[str], out_file: str):
    with open(out_file, 'w') as of:
        for x in d:
            of.write(f"{x}\n")


def read_in(in_file: str):
    j_list = []
    for line in open(in_file):
        j_list.append(json.loads(line))
    return j_list

def ccnet_wiki_filter(x):
    return x['ppl'] < 2287.064 and x['wiki_prob'] > 0.25


def iou(d1, d2):
    d1, d2 = (d1, d2) if len(d1) > len(d2) else (d2, d1)
    i = 0
    for x in d1:
        if x in d2:
            i += 1

    o = len(d1) + len(d2) - i
    return i / o, i / len(d2)


for line in open(filename):
    j_list.append(json.loads(line))


def f(j):
    a = j['bucket_mean'] >= 48 and j['bucket_std'] < 1
    poem = is_poem(j['text'])
    # b = j['ppl'] > 2287.064 and j['wiki_prob'] < 0.25
    # return a or b
    return a and not poem


fd = list(filter(f, j_list))
print(f"{len(fd)} / {len(j_list)} = {len(fd) / len(j_list)}")

not_fd = list(filter(lambda x: not f(x), j_list))

out_dir = 'split_out_b48'
os.makedirs(out_dir, exist_ok=True)
write_out(fd, os.path.join(out_dir, 'bad.jsonl'))
write_out(not_fd, os.path.join(out_dir, 'good.jsonl'))
