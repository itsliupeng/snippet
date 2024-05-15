import os

import json
import re

def read_in(in_file: str):
    j_list = []
    for line in open(in_file):
        j_list.append(json.loads(line))
    return j_list


def merge(q_file):
    q_list = read_in(q_file)
    result_list = []
    for x in q_list:
        ending = x['endings'][int(x['label'])]
        item = x['ctx']
        if ending.startswith(","):
            result_list.append(f"{item}{ending}")
        else:
            result_list.append(f"{item} {ending}")
    return result_list

d = merge("hellaswag_train.jsonl")


list(map(len, [d, l, m, s, xl, xs]))

data = list(set(d + l + m + s + xl + xs))

with open("winogrande.seed", "w") as of:
    for x in data:
        of.write(f"{x}\n")



