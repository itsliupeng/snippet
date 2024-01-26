import json
import re

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

# j_list = read_in("/ML-A100/home/liupeng/data/baidu_baike/new_d0913/baidubaike.json")
j_list = read_in("/ML-A100/home/liupeng/data/baidu_baike/d1025/baidubaike.json")


def merge_attribute(x: dict) -> str:
    if not isinstance(x, dict):
        return x

    s = ''
    for k, v in x.items():
        s += f"{k}:{v}，"

    if len(s) > 0:
        s = s[:-1]
    return s


def remove_link_tag(text):
    splits = text.split("\\n")
    if len(splits) == 0:
        return ""

    s = splits[0]
    hit_prev_link = False
    for x in splits[1:]:
        if len(x) <= 5:
            s = f"{s}{x}"
            hit_prev_link = True
        else:
            if hit_prev_link:
                s = f"{s}{x}"
            else:
                s = f"{s}\\n{x}"

            hit_prev_link = False
    return s

def merge_text(x) -> str:
    if not isinstance(x, list):
        x = remove_link_tag(x)
        x = x.replace("\\n", "\n")
        x = re.sub(r"\n{2,}", "\n", x)
        return x.strip()

    s = []
    for item_dict in x:
        for k, v in item_dict.items():
            if k.startswith("tags"):
                v = remove_link_tag(v)
                v = v.replace("\\n", "\n")
                v = re.sub(r"\n{2,}", "\n", v)
                s.append(f"{v.strip()}\n")
        s.append("\n")
    result = "".join(s)
    return result.strip()


def trans_digit(x):
    num = 0
    if x:
        try:
            num = int(x)
        except:
            num = 0
    return num


# def remove_duplicate_abstract(abstract, text):
#     # Replace the abstract in the text with an empty string
#     text = text.replace(abstract, "", 1)  # Only replace the first occurrence
#     return text.strip()


def remove_duplicate_abstract(abstract, text):
    # Replace the abstract in the text with an empty string
    text = text.replace(abstract, "")  # Only replace the first occurrence
    if len(abstract) > 0:
        return f"{abstract.strip()}\n{text.strip()}"
    else:
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
bracket_pattern = re.compile(
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
    if len(text) == 0:
        return text
    return bracket_pattern.sub('', text).strip()

def clean_newline(text):
    text = re.sub(r'\n[ \u3000]+\n', '\n\n', text) # 空格 和全角空格
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


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

black_words = [['小说', ['作品简介', '内容简介', '起点', '晋江', '潇湘书院', '创世中文', '红袖添香', '连城读书']],
               ['网络小说'], ['玄幻小说']]
def filter_black_word(text, black_words):
    hit_list = []
    for w in black_words:
        cur_hit_list = []
        for i in w:
            if isinstance(i, str):
                cur_hit_list.append(i in text)
            elif isinstance(i, list):
                cur_hit_list.append(any([j in text for j in i]))
            else:
                raise Exception(f'not supported format: {black_words}')

        hit_list.append(all(cur_hit_list))

    return not any(hit_list)

# len > 100
# 创建一个正则表达式模式，匹配所有的逗号和句号
punc_pattern = r'[，。；、]'
punc_regex = re.compile(punc_pattern)


def chinese_percentage(sentence):
    # 使用正则表达式找出所有中文字符
    chinese_characters = re.findall(r'[\u4e00-\u9fa5]', sentence)

    # 计算中文字符的占比
    percentage = (len(chinese_characters) / len(sentence)) * 100
    return percentage

def paragraph_miss_punctuation(t):
    if len(t) == 0:
        return False
    if chinese_percentage(t) < 0.8:
        return False
    splits = punc_regex.split(t)
    max_length = max(map(len, splits))

    return max_length >= 100

def text_miss_punctuation(text):
    splits = text.split("\n")
    return any([paragraph_miss_punctuation(x) for x in splits])

def mulu_filter(t):
    return "目录" in t
############################################################
# 1. get basic format
new_j_list = []
for j in j_list:
    new_j = {}
    new_j['_id'] = j.get('_id', "")
    new_j['title'] = j.get('标题', "")
    new_j['subtitle'] = j.get('副标题', "")
    new_j['abstract'] = j.get('摘要', "")
    new_j['attribute'] = merge_attribute(j.get('属性', ""))
    new_j['text'] = merge_text(j.get('正文', ""))
    # new_j['like'] = j['标题']
    new_j['edit'] = trans_digit(j.get('编辑次数', "0"))
    new_j['link'] = j.get('link', "")
    new_j_list.append(new_j)
print(f"{len(new_j_list)} / {len(j_list)}")

# 2. 正文摘要重复，[] 引用， 多个换行
for item_dict in new_j_list:
    clean_text = remove_duplicate_abstract(item_dict['abstract'], item_dict['text'])
    clean_text = clean_bracket_optimized(clean_text)
    clean_text = clean_newline(clean_text)
    item_dict['text'] = clean_text

print(f"{len(new_j_list)} / {len(j_list)}")

# 3. 小说
new_j_list = list(filter(lambda x: filter_black_word(x['text'], black_words), new_j_list))
no_novel_j_list = new_j_list
print(f"{len(new_j_list)} / {len(j_list)}")

# 标点符号
new_j_list = list(filter(lambda x: not text_miss_punctuation(x['text']), new_j_list))

print(f"{len(new_j_list)} / {len(j_list)}")

# 目录
new_j_list = list(filter(lambda x: not mulu_filter(x['text']), new_j_list))

print(f"{len(new_j_list)} / {len(j_list)}")



new_d = []
for item_dict in new_j_list:
    item_dict['content'] = item_dict['text']
    item_dict.pop('text')
    new_d.append(item_dict)


#####################
d1_titles = [f"{x['title']}_{x['subtitle']}" for x in d1]
d2_titles = [f"{x['title']}_{x['subtitle']}" for x in d2]

d1_content =  [f"{x['content']}" for x in d1]
d2_content =  [f"{x['content']}" for x in d2]
d1_count = {}
for x in d1_titles:
    if x not in d1_count:
        d1_count[x] = 0
    d1_count[x] += 1

list(filter(lambda a, b: b > 1, list(d1_count.items())))


d1_dict = dict([(x['content'], x) for x in d1])
d2_dict = dict([(x['content'], x) for x in d2])
