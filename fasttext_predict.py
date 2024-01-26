import pandas as pd
import jieba
import fasttext

d = pd.read_csv("zh_review_0804.csv", header=None)
d.columns = ['source', 'url', 'title', 'text']

def tokenize_chinese_text(text):
    """Tokenize Chinese text using jieba."""
    return " ".join(jieba.cut(text))

thunews_model = fasttext.load_model("/lp/data/THUCNews/THUCNews/thunews_model.bin")
def f(text):
    content = text.strip().replace('\n', '\\n')  # Escape newlines
    tokenized_content = tokenize_chinese_text(content)

    out = thunews_model.predict(tokenized_content, k=1)
    tag = out[0][0].split("__")[-1]
    prob = float(out[1][0])


    return tag, prob

dd = d.to_dict('records')

for item_dict in dd:
    tag, prob = f(item_dict['text'])
    item_dict['tag'] = tag
    item_dict['prob'] = prob


df = pd.DataFrame(dd)

