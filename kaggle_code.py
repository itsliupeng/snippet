import os


def get_file_lengths(directory):
    file_lengths = {}

    # 遍历指定目录下的所有文件
    for foldername, subfolders, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)

            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    file_lengths[file_path] = len(content)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    return file_lengths


# 使用函数
directory_path = '/ML-A100/home/liupeng/data/kaggle'  # 更改为你的目标目录路径
file_lengths = get_file_lengths(directory_path)
# for file, length in file_lengths.items():
#     print(f"{file}: {length} characters")

## lenght 10% 2548, 20% 4890
valid_file_lengths = {}
for file, length in file_lengths.items():
    if length > 4890.0:
        valid_file_lengths[file] = length


file_count = {}
for file, length in file_lengths.items():
    suffix = file.split(".")[-1]
    if suffix not in file_count:
        file_count[suffix] = 0

    file_count[suffix] += 1

"""
{'ipynb': 3305738,
 'py': 251041,
 'rmd': 92765,
 'r': 112147,
 'sql': 5529,
 'jl': 2810}
"""

r_file = []
py_file = []
rmd_file = []
notebook_file = []
file_count = {}
for file, length in valid_file_lengths.items():
    suffix = file.split(".")[-1]
    os.rename(file, os.path.join(suffix, os.path.basename(file)))

