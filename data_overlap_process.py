import json

def read_in(in_file: str):
    j_list = []
    for line in open(in_file):
        j_list.append(json.loads(line))
    return j_list

def write_out(d, out_file):
    with open(out_file, 'w') as of:
        for x in d:
            j = json.dumps(x, ensure_ascii=False)
            of.write(f"{j}\n")


d = read_in("filtered_scenario_data_new.jsonl")
selected_scenario = []
sum = 0
for item in d:
    scenario_key = item['scenario_key']
    class_name = scenario_key['scenario_spec']['class_name']
    subject = scenario_key['scenario_spec']['args']
    if 'subject' in subject:
        subject = subject['subject']
    else:
        subject = ""
    name = ".".join(class_name.split(".")[-2:])
    split = scenario_key['split']
    len_ins = len(item['instances'])
    if split == 'test' and len_ins > 0:
        print(f"{name}_{split}_{subject}", len_ins)
        selected_scenario.append(item)
        sum += len_ins

write_out(selected_scenario, "selected_scenario.jsonl")