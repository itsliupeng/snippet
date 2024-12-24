import json
from multiprocessing import Pool
from subprocess import CalledProcessError, run

import numpy as np
import torch
import webdataset as wds
import re

SAMPLE_RATE = 24000

from moshi.models import loaders

# 在模块级别定义全局变量


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

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def process_json_file(transcript_file, output_tar_prefix, gpu_id, start_line_no):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    # snacmodel = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()
    mimi_weight = "/lp/models/moshiko-pytorch-bf16/tokenizer-e351c8d8-checkpoint125.safetensors"
    mimi = loaders.get_mimi(mimi_weight, device='cuda')
    mimi.set_num_codebooks(8)  # up to 32 for mimi, but limited to 8 for moshi.

    output_tar = f"{output_tar_prefix}_{os.path.basename(transcript_file)}.tar"
    print(f"load model on gpu_id {gpu_id} with tar ${transcript_file}", flush=True)

    with torch.inference_mode():
        with wds.TarWriter(output_tar) as writer:
            count = 0
            try:
                with open(transcript_file, "r") as file:
                    # for byte_line in file:
                    for line in file:
                        if count < start_line_no:
                            continue
                        if True:
                            j = json.loads(line)
                            wav_file = j['wav']
                            wav_file = re.sub(r'(?<!_\d{8}_)\b' + '20241129' + r'\b', "20241129_20241202", wav_file, count=1)
                            text = j['refined']
                        try:
                            audio = load_audio(wav_file)[:SAMPLE_RATE * 163]  # 最长 40s
                            audio_name = "/".join(wav_file.split("/")[-4:])
                            speaker = "spotify"
                            audio = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).cuda()
                            codes = mimi.encode(audio).cpu().numpy()[0]
                        except Exception as e:
                            print(f"Exception {e} for line {line}")
                            continue

                        if True:
                            sample = {'text': text,
                                      'codec_label.npy': codes.tobytes(),
                                      'speaker': speaker,
                                      "__key__": audio_name}
                            writer.write(sample)

                            count += 1
                            if count % 100 == 0:
                                print(f"Processed {count} samples from {transcript_file} into {output_tar}")

            except Exception as e:
                print(f"Exception {e} for file {transcript_file}")


def write_webdataset(json_file_list, output_tar_prefix, num_processes, start_line_no):
    # Get the number of available CPU cores
    # num_workers = min(cpu_count(), len(json_file_list), 8*2)
    gpu_count = torch.cuda.device_count()
    print(f"num_processes: {num_processes}, start_line_no {start_line_no}")

    # Create a pool of workers
    with Pool(num_processes) as pool:
        pool.starmap(process_json_file,
                     [(json_file, output_tar_prefix, idx % gpu_count, start_line_no) for idx, json_file in enumerate(json_file_list)])


import argparse
import os
from multiprocessing import set_start_method

if __name__ == '__main__':
    # try:
    #     set_start_method('spawn')
    # except RuntimeError:
    #     pass  # 如果已经设置过启动方法，则忽略

    parser = argparse.ArgumentParser(
        description='Generate WebDataset from JSONL transcript files.'
    )

    parser.add_argument(
        '--transcript_data_dir',
        type=str,
        default='/lp/dataset/audio_data/librilight_processed/jsonl_splits',
        help='Directory containing JSONL transcript files.'
    )

    parser.add_argument(
        '--output_tar_prefix',
        type=str,
        default='webdataset/mimi_librilight',
        help='Output path and tar file prefix for the generated WebDataset.'
    )

    parser.add_argument(
        '--num_processes',
        type=int,
        default=16,
        help='Number of multiprocessing processes to use.'
    )

    parser.add_argument(
        '--start_line_no',
        type=int,
        default=0,
        help='Number of multiprocessing processes to use.'
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Set multiprocessing start method to 'spawn'
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass  # If the start method is already set, ignore the error

    # Extract arguments
    transcript_data_dir = args.transcript_data_dir
    output_tar_prefix = args.output_tar_prefix
    num_processes = args.num_processes
    start_line_no = args.start_line_no

    # Get list of JSON files in the current directory
    # transcript_data_dir = "/lp/dataset/audio_data/librilight_processed/jsonl_splits"
    ts_files = list(map(lambda x: os.path.join(transcript_data_dir, x), os.listdir(transcript_data_dir)))
    ts_files = sorted(ts_files)

    # Create output path and tar file prefix
    # output_tar_prefix = 'webdataset/librilight'

    # Call the function to generate WebDataset using multiprocessing
    write_webdataset(ts_files, output_tar_prefix, num_processes, start_line_no)
