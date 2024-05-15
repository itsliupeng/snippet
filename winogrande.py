import os

import json
import re

def read_in(in_file: str):
    j_list = []
    for line in open(in_file):
        j_list.append(json.loads(line))
    return j_list


def merge(q_file):
    # lines = open(label_file).readlines()
    # labels = list(map(lambda x: int(x.strip()), lines))
    q_list = read_in(q_file)
    result_list = []
    for x in q_list:
        answer = x[f'option{int(x["answer"])}']
        item = x['sentence'].replace("_", answer)
        result_list.append(item)
    return result_list

d = merge("train_debiased.jsonl")
l = merge("train_l.jsonl")
m = merge("train_m.jsonl")
s = merge("train_s.jsonl")
xl = merge("train_xl.jsonl")
xs = merge("train_xs.jsonl")

list(map(len, [d, l, m, s, xl, xs]))

data = list(set(d + l + m + s + xl + xs))

with open("winogrande.seed", "w") as of:
    for x in data:
        of.write(f"{x}\n")



