import os
import json

def merge_jsonl_files(input_dir, output_file):
    # 初始化一个空的列表来存储所有的 JSONL 行
    merged_data = []

    # 递归遍历目录
    for root, _, files in os.walk(input_dir):
        for file in files:
            # if file.endswith('.jsonl'):
            if file == "refined_transcription.jsonl":
                file_path = os.path.join(root, file)
                # print(f"Processing file: {file_path}")

                # 读取 JSONL 文件中的每一行并添加到列表
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            merged_data.append(data)
                        except json.JSONDecodeError:
                            print(f"Error decoding JSON in file: {file_path}")

    # 将所有 JSONL 数据写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in merged_data:
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Merged {len(merged_data)} JSONL lines into {output_file}")

# 使用
input_dir = '/lp/dataset/audio_data/librilight/split_whisper/large'
output_file = '/lp/dataset/audio_data/librilight/merged_refined_transcription.jsonl'
merge_jsonl_files(input_dir, output_file)
