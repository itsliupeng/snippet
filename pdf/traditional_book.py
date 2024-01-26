import os
import shutil

import re
import pandas as pd

import opencc
converter = opencc.OpenCC('t2s')
def tra_to_simple(text):
    simplified_text = converter.convert(text)
    return simplified_text


def is_chinese(text):
    # Regular expression pattern for matching texts that are solely Chinese characters
    pattern = r'[\u4e00-\u9fa5]+'
    return bool(re.search(pattern, text))



def find_dynasties(text):
    # 定义中国历代的一些主要朝代名称
    dynasties = ["汉", "唐", "宋", "元", "明", "清", "隋", "魏", "晋", "楚", "秦", "燕", "赵", "齐", "吴", "越", "商",
                 "周", "夏", "辽", "金", "南宋", "北宋", "东汉", "西汉", "东晋", "西晋", "东周", "西周"]
    # 构建正则表达式
    pattern = r'[\(\[]({})[\)\]]'.format("|".join(dynasties))
    # 使用正则表达式查找匹配的朝代
    matches = re.findall(pattern, text)
    return matches

def is_traditional_chinese_book(title):
    if not isinstance(title, str):
        # print(title)
        return False

    title = tra_to_simple(title)
    if not is_chinese(title):
        return False

    if len(find_dynasties(title)) > 0:
        return True
    elif ['四库全书'] in title:
        return False
    else:
        return False



def is_simply_chinese_book(title):
    if not isinstance(title, str):
        # print(title)
        return False

    title = tra_to_simple(title)
    if not is_chinese(title):
        return False

    if len(find_dynasties(title)) > 0:
        return False
    elif '四库全书' in title:
        return False
    else:
        return True


headers = ["Title", "Author", "Category", "Path"]
dtype_dict = {header: str for header in headers}
d = pd.read_csv("scanned_pdfs.csv", header=None, names=headers, dtype=dtype_dict)
other = d[(d['Category'] == 'other') | d['Category'].isna()]
other_list = other.to_dict(orient='records')
other_list = list(filter(lambda x: is_simply_chinese_book(x['Title']), other_list))

chn_smp = d[d['Category'] == 'chinese']
chn_trad = d[(d['Category'] == 'traditional_chinese') | (d['Category'] == 'traditional chinese')]
chn = pd.concat([chn_smp, chn_trad], ignore_index=True)

chn_list = chn.to_dict(orient='records')
chn_list = list(filter(lambda x: is_simply_chinese_book(x['Title']), chn_list))
all_chn_list = chn_list + other_list

chn_df = pd.DataFrame(all_chn_list)

title_list = chn_df['Title'].tolist()
path_list = chn_df['Path'].tolist()
dst_dir = "zh_book_A"
for t, p in zip(title_list, path_list):
    if not os.path.exists(p):
        continue

    file_size = os.path.getsize(p) / (1024 ** 3)
    if file_size > 0.3 or file_size < 0.1:
        continue
    basename = os.path.basename(p)
    if not basename.endswith(".pdf"):
        basename = f"{basename}.pdf"
    dst_name = os.path.join(dst_dir, basename)
    shutil.copyfile(p, dst_name)




####
cadidate_category_list = ['english-chinese ', 'mandarinchinese', 'chinese,english', 'chinese, english', 'english, chinese', 'english,chinese',
'english-chinese', 'chinese/english', 'chinese,_english', 'chinese-english',
 'chinese_','chinese', 'english/chinese', 'chinese ', 'chinese,']

chn_list = []
for x in cadidate_category_list:
    chn_list.append(d[d['Category'].str.lower() == x])

from functools import reduce

chn = reduce(lambda x, y: pd.concat([x, y], ignore_index=True), chn_list)
# en = d[d['Category'] == 'english']
