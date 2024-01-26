import json
import re

def read_in(in_file: str):
    j_list = []
    for line in open(in_file):
        j_list.append(json.loads(line))
    return j_list


d = read_in("baike_final.jsonl")


def remove_duplicate_abstract(abstract, text):
    # Replace the abstract in the text with an empty string
    text = text.replace(abstract, "", 1)  # Only replace the first occurrence
    return text.strip()

#
# def clean_bracket(text):
#     # 删除 [数字-数字] 模式
#     text = re.sub(r'\[\d+\]', '', text)
#     text = re.sub(r'\[\d+-\d+\]', '', text)
#     #比如 汉字[1.0]
#     text = re.sub(r'(?<=[\u4e00-\u9fa5])\s*\[\d+\.?\d*\]', '', text)
#     # 删除 [ISSN:后面跟一系列的字符,再跟一系列的字符] 模式
#     text = re.sub(r'\[ISSN:[^,]+,[^\]]+\]', '', text)
#     # 删除以标点符号后面紧跟 [ 开始的模式
#     text = re.sub(r'[.,!?;，。！？]\s*\[[^\]]+\]', '', text)
#
#     # 百科关键词
#     text = re.sub("\[编辑\]", "", text)
#     text = re.sub("\[注\]", "", text)
#     text = re.sub("\[\s*\]", "", text)
#
#     return text.strip()


# 预编译正则表达式
pattern = re.compile(
    r'\[\d+\]'                            # [数字]
    r'|\[\d+-\d+\]'                      # [数字-数字]
    r'|(?<=[\u4e00-\u9fa5])\s*\[\d+\.?\d*\]'  # 汉字[数字(可能为小数)]
    r'|\[ISSN:[^,]+,[^\]]+\]'            # [ISSN:...,...]
    r'|[.,!?;，。！？]\s*\[[^\]]+\]'      # 标点[...]
    r'|\[编辑\]'                         # [编辑]
    r'|\[注\]'                           # [注]
    r'|\[\s*\]'                          # [空格]
)

def clean_bracket_optimized(text):
    return pattern.sub('', text).strip()

def clean_newline(text):
    text = re.sub(r'\n[ \u3000]+\n', '\n\n', text) # 空格 和全角空格
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

for item_dict in d:
    clean_text = remove_duplicate_abstract(item_dict['abstract'], item_dict['text'])
    clean_text = clean_bracket_optimized(clean_text)
    clean_text = clean_newline(clean_text)
    item_dict['text'] = clean_text


def write_out(d, out_file):
    with open(out_file, 'w') as of:
        for x in d:
            j = json.dumps(x, ensure_ascii=False)
            of.write(f"{j}\n")

write_out(d, 'baike_final_d0811.jsonl')


new_d = []
for item_dict in d:
    item_dict['content'] = item_dict['text']
    item_dict.pop('text')
    new_d.append(item_dict)

