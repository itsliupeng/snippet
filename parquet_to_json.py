import os
import json
import pandas as pd

cur_dir = "."
all_text = []
for i in os.listdir(cur_dir):
    d = pd.read_parquet(os.path.join(cur_dir, i), engine='pyarrow')
    text = list(d['text'])
    all_text += text

with open('all.jsonl', 'w') as of:
    for i in all_text:
        j = {'content': i}
        of.write(f"{json.dumps(j, ensure_ascii=False)}\n")


