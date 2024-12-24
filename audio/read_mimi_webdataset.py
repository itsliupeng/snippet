
import webdataset as wds
from snac import SNAC

import os
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ.pop('SLURM_PROCID', None)

# Define the path to the .tar file
# dataset_path = "/lp/pretrain_audio_data/webdataset/mimi_librilight/mimi_librilight_part_15.jsonl.tar"
dataset_path = "spotify_1st_phase_split_375.jsonl.tar"
dataset = wds.WebDataset(dataset_path)
raw = next(iter(dataset))

import torch
import numpy as np
codec_label = torch.from_numpy(np.frombuffer(raw['mp3.codec_label.npy'], dtype=np.int64)).view(8, -1)


#########################
from moshi.models import loaders, LMGen

mimi_weight = "/lp/models/moshiko-pytorch-bf16/tokenizer-e351c8d8-checkpoint125.safetensors"
mimi_model = loaders.get_mimi(mimi_weight, device='cpu')


with torch.inference_mode():
    audio_hat = mimi_model.decode(codec_label.unsqueeze(0))[0]

import soundfile as sf
sf.write("test_mimi.wav", audio_hat.squeeze().cpu().numpy(), 24000)

######################################################################

import webdataset as wds
from collections import Counter
import json
import numpy as np

# 定义 .tar 文件路径
dataset_path = "spotify_1st_phase_split_375.jsonl.tar"

# 加载数据集
dataset = wds.WebDataset(dataset_path)

# 初始化计数器
count_list = []
text_list = []

# 遍历数据集
for sample in dataset:
    if 'mp3.text' in sample:
        # 解码并计算长度
        text = sample['mp3.text'].decode('utf-8')  # 解码为字符串
        text_list.append(text)
        # 更新长度分布
        count_list.append(len(text.split()))
    else:
        print("Sample does not contain 'mp3.text'")

count_list = np.array(count_list)