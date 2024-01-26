import json
import random

d = json.load(open("dataset.json"))


import re
def is_numbers_operators_and_punctuation(s):
    # 正则表达式匹配数字、+、-、*、/符号、空格和常见的标点符号
    pattern = r'^[\d+\-*/\s]+$'
    return bool(re.match(pattern, s))


instruction_prefix = [
    ("Can you provide the answer to", "?"),
    ("Solve for the value of", "."),
    ("What is the value obtained from", "?"),
    ("Can you calculate", "?"),
    ("Show me the result of the following problem:", "."),
    ("What is the outcome of evaluating", "?"),
    ("Show the step-by-step method for evaluating", "."),
    ("Find the numerical outcome of", "."),
]

text_list = []
for x in d:
    instruction = x['instruction']
    if is_numbers_operators_and_punctuation(instruction):
        idx = random.randint(0, len(instruction_prefix)-1)
        prefix = instruction_prefix[idx]
        instruction = f"{prefix[0]} {instruction}{prefix[1]}"

    if x['output'].startswith(x['input']):
        step_text = x['output']
    else:
        step_text = f"{x['input']} =\n{x['output']}"

    s = f"{instruction}\n{step_text}.\nThe answer is {x['answer']}."
    text_list.append(s)

# for x in text_list:
#     print(x)
#     print(f"\n---------------\n\n")

with open("goat_arithmetic_text.txt", "w") as of:
    for x in text_list:
        of.write(f"{x}\n\n")

random.shuffle(text_list)
with open("goat_arithmetic_text_shuffle.txt", "w") as of:
    for x in text_list:
        of.write(f"{x}\n\n")

def write_out(d, out_file):
    with open(out_file, 'w') as of:
        for x in d:
            j = json.dumps(x, ensure_ascii=False)
            of.write(f"{j}\n")

text_json_list = [{'id': idx, 'content': x} for idx, x in enumerate(text_list)]