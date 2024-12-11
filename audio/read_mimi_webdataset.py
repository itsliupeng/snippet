
import webdataset as wds
from snac import SNAC

import os
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ.pop('SLURM_PROCID', None)

# Define the path to the .tar file
# dataset_path = "/lp/pretrain_audio_data/webdataset/mimi_librilight/mimi_librilight_part_15.jsonl.tar"
dataset_path = "mimi_quora_xttsv2_part_ac.tar"
dataset = wds.WebDataset(dataset_path)
raw = next(iter(dataset))

import torch
import numpy as np
codec_label = torch.from_numpy(np.frombuffer(raw['wav.codec_label.npy'], dtype=np.int64)).view(8, -1)


#########################
from moshi.models import loaders, LMGen

mimi_weight = "/lp/models/moshiko-pytorch-bf16/tokenizer-e351c8d8-checkpoint125.safetensors"
mimi_model = loaders.get_mimi(mimi_weight, device='cpu')


with torch.inference_mode():
    audio_hat = mimi_model.decode(codec_label.unsqueeze(0))[0]

import soundfile as sf
sf.write("test_mimi.wav", audio_hat.squeeze().cpu().numpy(), 24000)