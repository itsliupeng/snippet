from collections import defaultdict
import ast

# A basic tokenizer for demonstrationsp
def simple_tokenizer(text):
    return text.split()

# Generate n-grams from tokens
def generate_ngrams(tokens, n):
    return zip(*[tokens[i:] for i in range(n)])

def compute_token_overlap_simplified(instance_str, overlapping_ngram_counts, frequency=0):
    tokens = simple_tokenizer(instance_str)
    ngram_counts_dict = defaultdict(int)

    # 构建 ngram 到计数的字典
    for ngram, count in overlapping_ngram_counts.items():
        ngram = tuple(ast.literal_eval(ngram))
        ngram_counts_dict[ngram] = count

    total_token_count = 0
    counts = 0

    # 遍历 n-grams
    for ngram in generate_ngrams(tokens, 2):  # 假设 n-gram 的大小为 2
        if ngram_counts_dict[ngram] != 0:
            counts += 1
        total_token_count += 1

    unweighted_score = counts / total_token_count if total_token_count > 0 else 0

    return unweighted_score


# Test cases
test_cases = [
    ("the quick brown fox jumps over the lazy dog", {str(("the", "quick")): 2, str(("lazy", "dog")): 1}, 0),
    ("lorem ipsum dolor sit amet, consectetur adipiscing elit", {str(("lorem", "ipsum")): 3, str(("dolor", "sit")): 2, str(("adipiscing", "elit")): 1}, 2),
    ("example of a very simple text for testing", {str(("very", "simple")): 2, str(("text", "for")): 3, str(("example", "of")): 1}, 1)
]

# Running the test cases
results = [compute_token_overlap_simplified(*case) for case in test_cases]
print(results)


def extract(lds_str):
    # 从字典中提取字符串
    lds_str = lds_str['lds']

    # 提取Pid
    pid_start = lds_str.find('Pid: ') + len('Pid: ')
    pid_end = lds_str.find(',', pid_start)
    pid = int(lds_str[pid_start:pid_end])

    # 提取Topic
    topic_start = lds_str.find('Topic: ', pid_end) + len('Topic: ')
    topic_end = lds_str.find(',', topic_start)
    topic = int(lds_str[topic_start:topic_end])

    # 提取Words
    words_start = lds_str.find('Words: ', topic_end) + len('Words: ')
    words = lds_str[words_start:]

    return (pid, topic, words)



import torch
import os
files = os.listdir('.')
for x in files[:]:
    file_path = os.path.join(x, "model_optim_rng.pt")
    print(file_path)
    m = torch.load(file_path, "cpu")
    m['opt_param_scheduler']['num_steps'] = 44000 * 1024
    torch.save(m, file_path)



