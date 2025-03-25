
import webdataset as wds

import os
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ.pop('SLURM_PROCID', None)

dataset_path = "/gpfs/public/pretrain/data/audio/VoiceAssistant-400K/cosyvoice_voice_assistant_200k_en/_valid_meta_part_27.tar"
dataset = wds.WebDataset(dataset_path)
# raw = next(iter(dataset))

raw = None
hit_count = 0
count = 0
for item in iter(dataset):
    raw = item
    if count == hit_count:
        break
    else:
        count += 1

import torch
import numpy as np
codec_label=torch.from_numpy(np.frombuffer(raw['answer_audio'], dtype=np.int32))


#########################

import os
import gzip
import json
import time
from tqdm import tqdm
import numpy as np
import soundfile as sf  # For saving audio files
# import torchaudio
import sys
import onnxruntime
import torch
import whisper

"""
# cd /gpfs/public/pretrain/liupeng/code/CosyVoice

ROOT_DIR = os.getcwd()
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

cosyvoice = CosyVoice2('/lp/models/CosyVoice2-0.5B', load_jit=False, load_onnx=False, load_trt=False).model
"""

speech_token = codec_label.unsqueeze(0)
prompt_token = torch.zeros(1, 0, dtype=speech_token.dtype)
prompt_feat = torch.zeros(1, 0, 80)
embedding = torch.zeros(1, 192)

tts_mel, _ = cosyvoice.flow.inference(token=speech_token.to(device),
                                      token_len=torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(device),
                                      prompt_token=prompt_token.to(device),
                                      prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(
                                          device),
                                      prompt_feat=prompt_feat.to(device),
                                      prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(
                                          device),
                                      embedding=embedding.to(device),
                                      finalize=True)

hift_cache_source = torch.zeros(1, 1, 0)
tts_speech, tts_source = cosyvoice.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
tts_speech = tts_speech.cpu()

torchaudio.save(f'zero_shot_{hit_count}.wav', tts_speech, 24000)

print("Done")