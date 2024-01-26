import json
import numpy as np
import random


def print_percentile(len_list, num_splits=10):
    len_list = np.array(len_list)
    # Define percentiles
    percentiles = list(map(int, np.linspace(0, 100, num_splits + 1)))
    # Calculate percentile values
    percentile_values = np.percentile(len_list, percentiles)
    # Print percentile distribution
    for percentile, value in zip(percentiles, percentile_values):
        print(f"Percentile {percentile}: {value}")


def split_parts(data, key, metric_list, num=3):
    metric_list = np.array(metric_list)
    # Define percentiles
    percentiles = list(map(int, np.linspace(0, 100, num + 1)))
    # Calculate percentile values
    percentile_values = np.percentile(metric_list, percentiles)
    result_list = []
    for s, e in zip(percentile_values[:-1], percentile_values[1:]):
        result_list.append(list(filter(lambda x: s <= x[key] < e, data)))
    return result_list


def sample_N(data, num):
    # not support dict item.
    if len(data) == 0:
        return []

    p = num / len(data)
    results = set()
    for x in data:
        if random.random() < p:
            results.add(x)
    while len(results) < num:
        idx = random.randint(0, len(data))
        results.add(data[idx])
    return list(results)[0:num]


# wiki
in_file = "dedup_all_baichuan_13b_w5m200.jsonl"
keyword = ''

j_list = []
for line in open(in_file):
    try:
        j = json.loads(line)
        j_list.append(j)
    except Exception as e:
        print(e)

# 长度分布
len_list = list(map(lambda x: len(x['text']), j_list))
print_percentile(len_list)

# ppl 分布
vj_list = list(filter(lambda x: len(x['text']) > 88, j_list))
llm_ppl = list(map(lambda x: x['baichuan_13b_ppl'], vj_list))
cc_ppl = list(map(lambda x: x['ppl'], vj_list))
views = list(map(lambda x: x['view'], vj_list))

print_percentile(llm_ppl)
print_percentile(cc_ppl)
print_percentile(views)

# 根据 ppl 10 划分
vj_list_A = list(filter(lambda x: x['baichuan_13b_ppl'] <= 10, vj_list))
vj_list_B = list(filter(lambda x: x['baichuan_13b_ppl'] > 10, vj_list))

# vj_list_A 中 cc_ppl, views 分布
llm_ppl_A = list(map(lambda x: x['baichuan_13b_ppl'], vj_list_A))
cc_ppl_A = list(map(lambda x: x['ppl'], vj_list_A))
views_A = list(map(lambda x: x['view'], vj_list_A))
print_percentile(llm_ppl_A)
print_percentile(cc_ppl_A)
print_percentile(views_A)

llm_ppl_B = list(map(lambda x: x['baichuan_13b_ppl'], vj_list_B))
cc_ppl_B = list(map(lambda x: x['ppl'], vj_list_B))
views_B = list(map(lambda x: x['view'], vj_list_B))
print_percentile(llm_ppl_B)
print_percentile(cc_ppl_B)
print_percentile(views_B)

r_list = split_parts(vj_list, key='baichuan_13b_ppl', metric_list=llm_ppl, num=10)
print(list(map(len, r_list)))

for idx, part in enumerate(r_list):
    part_sample = sample_N(list(map(lambda x: json.dumps(x, ensure_ascii=False), part)), 1000)
    with open(f"part_{idx}.json", 'w') as of:
        for x in part_sample:
            of.write(f"{x}\n")

