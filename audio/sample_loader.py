import json
import torch
import numpy as np
import whisper

def get_mel(audio):
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio)
    return mel

def sample_loader(raw: dict) -> dict:
    data = json.loads(raw["metadata.json"])
    question = data['question']
    answer = data['answer']
    audio = np.frombuffer(raw['wav.npy'], dtype=np.int16).astype(np.float32) / 32768.0
    mel = get_mel(audio)
    duration_ms = (len(audio) / 16000) * 1000
    length = int(duration_ms / 20) + 1
    codec_label = np.frombuffer(raw['snack_label.npy'], dtype=np.int16).reshape(-1, 7).astype(np.int32)
    return dict(
        __key__=raw["__key__"],
        mel=mel,
        length=length,
        codec_label=torch.from_numpy(codec_label),
        question=question,
        answer=answer
    )

def part_filter(part: str) -> bool:
    return True

############################################################

import json
import torch
import numpy as np
import whisper

def get_mel(audio):
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio)
    return mel

def sample_loader(raw: dict) -> dict:
    data = raw
    question = data['question']
    answer = data['answer']
    audio = np.frombuffer(raw['question_audio'], dtype=np.int16).astype(np.float32) / 32768.0
    mel = get_mel(audio)
    duration_ms = (len(audio) / 16000) * 1000
    length = int(duration_ms / 20) + 1
    codec_label = np.frombuffer(raw['answer_audio'], dtype=np.int16).reshape(-1, 7).astype(np.int32)
    return dict(
        __key__=raw["__key__"],
        mel=mel,
        length=length,
        codec_label=torch.from_numpy(codec_label),
        question=question,
        answer=answer
    )

def part_filter(part: str) -> bool:
    return True

#########################
import json
import torch
import numpy as np
import whisper

def get_mel(audio):
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio)
    return mel

def sample_loader(raw: dict) -> dict:
    data = raw
    question = data['question']
    answer = data['answer']
    audio = np.frombuffer(raw['question_audio'], dtype=np.int16).astype(np.float32) / 32768.0
    mel = get_mel(audio)
    duration_ms = (len(audio) / 16000) * 1000
    length = int(duration_ms / 20) + 1
    codec_label = np.frombuffer(raw['answer_audio'], dtype=np.int32).reshape(-1, 7).astype(np.int32)
    return dict(
        __key__=raw["__key__"],
        mel=mel,
        length=length,
        codec_label=torch.from_numpy(codec_label).T, # (seqlen, 7)
        question=question,
        answer=answer
    )

def part_filter(part: str) -> bool:
    return True