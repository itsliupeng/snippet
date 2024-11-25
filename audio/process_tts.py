import webdataset as wds
import json
import os
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Optional, Union
import numpy as np
import io
import base64
from multiprocessing import Pool, cpu_count
from collections import defaultdict
import whisper
import soundfile as sf
import resampy
import torch
from snac import SNAC

SAMPLE_RATE = 24000


import torchaudio
import torchaudio.transforms as T

def load_audio_torchaudio(file, target_sr=SAMPLE_RATE):
    waveform, original_sr = torchaudio.load(file)

    # Convert to mono if it's multi-channel
    if waveform.shape[0] > 1:  # Check if more than one channel
        waveform = torch.mean(waveform, dim=0, keepdim=True)  # Downmix to mono

    if original_sr != target_sr:
        resampler = T.Resample(orig_freq=original_sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform

def load_audio_soundfile(file: str, sr: int = SAMPLE_RATE):
    # Read the audio file
    data, original_sr = sf.read(file)

    # Resample if necessary
    if original_sr != sr:
        data = resampy.resample(data, original_sr, sr)

    return data

def load_audio(file: str, sr: int = SAMPLE_RATE):
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten()


def deconstruct_tensor(codes):
    assert len(codes) == 3
    codes_len = codes[0].size(-1)
    codes_1 = codes[0].cpu().view(1, 1, codes_len)
    codes_2 = codes[1].cpu().view(1, 2, codes_len)
    codes_3 = codes[2].cpu().view(1, 4, codes_len)
    codes_tensor = torch.cat([
        codes_1[:, 0],
        codes_2[:, 0],
        codes_3[:, 0],
        codes_3[:, 1],
        codes_2[:, 1],
        codes_3[:, 2],
        codes_3[:, 3]
    ], dim=0)
    return codes_tensor

def process_json_file(transcript_file, output_tar_prefix, gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    snacmodel = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()
    output_tar = f"{output_tar_prefix}_{os.path.basename(transcript_file)}.tar"
    with torch.inference_mode():
        with wds.TarWriter(output_tar) as writer:
            count = 0
            for line in open(transcript_file):
                j = json.loads(line.strip())
                wav_file = j['wav']
                text = j['refined']
                audio = load_audio_torchaudio(wav_file)
                audio_name = "/".join(wav_file.split("/")[-3:])

                audio = audio.unsqueeze(0).cuda()
                codes = snacmodel.encode(audio)
                codes_tensor = deconstruct_tensor(codes)

                sample = {'text': text,
                          'codes_label.npy': codes_tensor.numpy().tobytes(),
                          "__key__": audio_name}
                writer.write(sample)

                count += 1
                if count % 100 == 0:
                    print(f"Processed {count} samples from {transcript_file} into {output_tar}")

def write_webdataset(json_file_list, output_tar_prefix):
    # Get the number of available CPU cores
    num_workers = min(cpu_count(), len(json_file_list))
    print(f"num_workers: {num_workers}")

    # Create a pool of workers
    with Pool(num_workers) as pool:
        pool.starmap(process_json_file, [(json_file, output_tar_prefix, idx % 8) for idx, json_file in enumerate(json_file_list)])


# Get list of JSON files in the current directory
transcript_data_dir = "/lp/dataset/audio_data/librilight/split_whisper/jsonl_splits"
ts_files = list(map(lambda x: os.path.join(transcript_data_dir, x), filter(lambda x: x.endswith(".jsonl"), os.listdir(transcript_data_dir))))
ts_files = sorted(ts_files)

# Create output path and tar file prefix
output_tar_prefix = 'webdataset/librilight'

# Call the function to generate WebDataset using multiprocessing
write_webdataset(ts_files, output_tar_prefix)
