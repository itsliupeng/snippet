import json
import copy

d = json.load(open("question.json"))

question_l = []
for x in d:
    question_l.append(f"{x['title']}\n{x['answer2']}")


import pandas as pd
d = pd.read_csv("地理.csv")
dili_l = d['text'].tolist()


d = pd.read_csv("高中数学_1.csv")
math_l = d['text'].tolist()

d = pd.read_csv("高中语文.csv")
languge_l = d['text'].tolist()


all_l = question_l + dili_l + math_l + languge_l # 51200




##############################################
d = read_in("selected_scenario.jsonl")

dd = d[0]
scenario_list = []
for k, v in zip(["all", "geography", "math", "language"], [all_l, dili_l, math_l, languge_l]):
    dd['scenario_key']['scenario_spec']['class_name'] = f"choices_{k}"
    dd['scenario_key']['scenario_spec']["args"] = {}

    instances = []
    for idx, x in enumerate(v):
        instances.append({
            'input': x,
            'references': [],
            'id': f"id{idx}"
        })
    dd['instances'] = instances
    scenario_list.append(copy.deepcopy(dd))

list(map(lambda x: (x['scenario_key']['scenario_spec'], len(x['instances'])), scenario_list))


def write_out(d, out_file):
    with open(out_file, 'w') as of:
        for x in d:
            j = json.dumps(x, ensure_ascii=False)
            of.write(f"{j}\n")

write_out(scenario_list, "choices_scenario.jsonl")