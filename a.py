import os, json

files= os.listdir(".")

for file in files:
    d = json.load(open(file))
    contents = list(map(lambda x: f"{x['question']}\nanswer: {x['answer']}", d))
    with open(file + ".jsonl", "w") as f:
        for x in contents:
            item = {"content": x}
            j = json.dumps(item, ensure_ascii=False)
            f.write(f"{j}\n")



