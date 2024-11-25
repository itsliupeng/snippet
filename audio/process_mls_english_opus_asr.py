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

SAMPLE_RATE = 16000


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


def process_json_file(transcript_file, output_tar_prefix):
    output_tar = f"{output_tar_prefix}_{os.path.basename(transcript_file)}.tar"
    with wds.TarWriter(output_tar) as writer:
        count = 0
        for line in open(transcript_file):
            sp = line.strip().split("\t")
            if (len(sp) != 2):
                print(f"{line} format error")
                continue
            audio_name, text = sp

            audio_name_sp = audio_name.split("_")
            audio_path = f'audio/{"/".join(audio_name_sp[:-1])}/{audio_name}.opus'
            if not os.path.exists(audio_path):
                print(f"{audio_path} not exist.")
                continue

            audio_data = load_audio(audio_path)

            sample = {"wav.npy": audio_data.tobytes(),
                      "text": text,
                      "__key__": audio_name}
            writer.write(sample)
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} samples from {transcript_file} into {output_tar}")


def write_webdataset(json_file_list, output_tar_prefix):
    # Get the number of available CPU cores
    num_workers = min(cpu_count(), len(json_file_list)) // 4
    print(f"num_workers: {num_workers}")

    # Create a pool of workers
    with Pool(num_workers) as pool:
        pool.starmap(process_json_file, [(json_file, output_tar_prefix) for json_file in json_file_list])


# Get list of JSON files in the current directory
transcript_data_dir = "/lp/dataset/audio_data/mls_english_opus/train/transcripts_splits"
ts_files = list(map(lambda x: os.path.join(transcript_data_dir, x), filter(lambda x: x.endswith(".txt"), os.listdir(transcript_data_dir))))
ts_files = sorted(ts_files)

# Create output path and tar file prefix
output_tar_prefix = 'mls_english'

# Call the function to generate WebDataset using multiprocessing
write_webdataset(ts_files, output_tar_prefix)
