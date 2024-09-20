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

SAMPLE_RATE = 16000


def load_audio(audio_bytes: str, sr: int = SAMPLE_RATE):
    audio_bytes = base64.b64decode(audio_bytes)
    byte_stream = io.BytesIO(audio_bytes)
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "4",
        "-i", "pipe:0",
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, input=byte_stream.read(), capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten()


def process_json_file(json_file, output_tar_prefix):
    output_tar = f"{output_tar_prefix}_{os.path.basename(json_file).split('.')[0]}.tar"
    with wds.TarWriter(output_tar) as writer:
        count = 0
        for line in open(json_file):
            item = json.loads(line.strip())
            audio = load_audio(item['question_audio']['bytes'])
            sample = {
                '__key__': f"{item['index']}_{item['round']}",
                'question': item['question'],
                'question_audio': audio.tobytes(),
                'answer': item['answer'],
                'answer_snac': item['answer_snac'],
                'index': item['index'],
                'round': item['round'],
                'split_name': item['split_name']
            }
            writer.write(sample)
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} samples from {json_file} into {output_tar}")


def write_webdataset(json_file_list, output_tar_prefix):
    # Get the number of available CPU cores
    num_workers = min(cpu_count(), len(json_file_list)) // 4
    print(f"num_workers: {num_workers}")

    # Create a pool of workers
    with Pool(num_workers) as pool:
        pool.starmap(process_json_file, [(json_file, output_tar_prefix) for json_file in json_file_list])


# Get list of JSON files in the current directory
json_data_dir = "/gpfs/public/dataset/audio_data/VoiceAssistant-400K/json"
json_files = list(map(lambda x: os.path.join(json_data_dir, x), filter(lambda x: x.endswith(".json"), os.listdir(json_data_dir))))
json_files = sorted(json_files)

# Create output path and tar file prefix
output_tar_prefix = 'voice_assistant_400k'

# Call the function to generate WebDataset using multiprocessing
write_webdataset(json_files, output_tar_prefix)
