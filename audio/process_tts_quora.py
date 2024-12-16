import argparse
import os
import re
from multiprocessing import Pool
from multiprocessing import set_start_method
from subprocess import CalledProcessError, run

import numpy as np
import torch
import webdataset as wds
from snac import SNAC

SAMPLE_RATE = 24000

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


def deconstruct_tensor(codes):
    """
    :param codes:
    :return: shape [7, -1]
    """
    assert len(codes) == 3
    codes_len = codes[0].size(-1)
    codes_1 = codes[0].cpu().view(1, codes_len, 1)
    codes_2 = codes[1].cpu().view(1, codes_len, 2)
    codes_3 = codes[2].cpu().view(1, codes_len, 4)
    codes_tensor = torch.cat([
        codes_1[..., 0],
        codes_2[..., 0],
        codes_3[..., 0],
        codes_3[..., 1],
        codes_2[..., 1],
        codes_3[..., 2],
        codes_3[..., 3]
    ], dim=0)
    return codes_tensor

def process_json_file(transcript_file, output_tar_prefix, gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    snacmodel = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()
    output_tar = f"{output_tar_prefix}_{os.path.basename(transcript_file)}.tar"
    print(f"load model on gpu_id {gpu_id} with tar ${transcript_file}", flush=True)
    with torch.inference_mode():
        with wds.TarWriter(output_tar) as writer:
            count = 0
            try:
                with open(transcript_file, "rb") as file:
                    for byte_line in file:
                        try:
                            try:
                                line = byte_line.decode('utf-8', errors='strict')
                            except UnicodeDecodeError as e:
                                print(f"Decoding failed: {e}")
                                continue

                            j = line.strip().split("\t")
                            if len(j) != 2 or j[0] == 'text': # header
                                continue

                            wav_file = j[1]
                            text = j[0]
                            # audio = load_audio_torchaudio(wav_file)
                            audio = load_audio(wav_file)[:SAMPLE_RATE*40]  # 最长 40s

                            audio_name = "/".join(wav_file.split("/")[-4:])

                            match = re.search(r'quora_xttsv2_([a-zA-Z]+_[a-zA-Z]+)/', wav_file)
                            if match:
                                xx_xx = match.group(1)  # 获取匹配的 xx_xx
                                speaker = xx_xx.replace('_', ' ')  # 将 "_" 替换为 " "
                            else:
                                speaker = "cosyvoice_中文女"


                            audio = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).cuda()
                            codes = snacmodel.encode(audio)
                            codes_tensor = deconstruct_tensor(codes)

                            sample = {'text': text,
                                      'codec_label.npy': codes_tensor.numpy().tobytes(),
                                      'speaker': speaker,
                                      "__key__": audio_name}
                            writer.write(sample)

                            count += 1
                            if count % 1000 == 0:
                                print(f"Processed {count} samples from {transcript_file} into {output_tar}")
                        except Exception as e:
                            print(f"Exception {e} for line {line}")
                            continue
            except Exception as e:
                print(f"Exception {e} for file {transcript_file}")

def write_webdataset(json_file_list, output_tar_prefix, num_processes):
    # Get the number of available CPU cores
    # num_workers = min(cpu_count(), len(json_file_list), 8*2)
    print(f"num_processes: {num_processes}")

    # Create a pool of workers
    with Pool(num_processes) as pool:
        pool.starmap(process_json_file, [(json_file, output_tar_prefix, idx % 8) for idx, json_file in enumerate(json_file_list)])


if __name__ == '__main__':
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass  # 如果已经设置过启动方法，则忽略

    parser = argparse.ArgumentParser(
        description='Generate WebDataset from JSONL transcript files.'
    )

    parser.add_argument(
        '--transcript_data_dir',
        type=str,
        default='/lp/data/sythetic_audio/quora/quora_xttsv2_meta_splits',
        help='Directory containing JSONL transcript files.'
    )

    parser.add_argument(
        '--output_tar_prefix',
        type=str,
        default='webdataset/quora_xttsv2/quora_xttsv2',
        help='Output path and tar file prefix for the generated WebDataset.'
    )

    parser.add_argument(
        '--num_processes',
        type=int,
        default=16,
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

    # Get list of JSON files in the current directory
    # transcript_data_dir = "/lp/dataset/audio_data/librilight_processed/jsonl_splits"
    ts_files = list(map(lambda x: os.path.join(transcript_data_dir, x),  os.listdir(transcript_data_dir)))
    ts_files = sorted(ts_files)

    # Create output path and tar file prefix
    # output_tar_prefix = 'webdataset/librilight'

    # Call the function to generate WebDataset using multiprocessing
    write_webdataset(ts_files, output_tar_prefix, num_processes)
