import argparse
import os
import re
from multiprocessing import Pool
from multiprocessing import set_start_method
from subprocess import CalledProcessError, run

import numpy as np
import torch
import webdataset as wds
import onnxruntime
import whisper
import math

speech_tokenizer_model = "/lp/models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx"

def extract_speech_token(speech_tokenizer_session, file, sr=16000):
    speech = whisper.load_audio(file, sr=sr)
    duration = min(speech.shape[-1] / sr, 30)
    speech = whisper.pad_or_trim(speech)
    speech = np.expand_dims(speech, 0)
    # assert speech.shape[1] / 16000 <= 30, 'do not support extract speech token for audio longer than 30s'
    feat = whisper.log_mel_spectrogram(speech, n_mels=128)

    speech_token = speech_tokenizer_session.run(None,
                                                {speech_tokenizer_session.get_inputs()[0].name: feat.numpy(),
                                                speech_tokenizer_session.get_inputs()[1].name:
                                                np.array([feat.shape[2]], dtype=np.int32)})[0]

    tokens_len = math.ceil(duration * 25)
    return speech_token[:, :tokens_len]


def process_json_file(transcript_file, output_tar_prefix, gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    option = onnxruntime.SessionOptions()
    option.intra_op_num_threads = 1
    # option.enable_mem_pattern = True  # Optimize memory patterns
    # option.enable_cpu_mem_arena = True  # Enable memory arena (useful for large models)
    option.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL  # Parallel execution
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    speech_tokenizer_session = onnxruntime.InferenceSession(speech_tokenizer_model, sess_options=option,
                                                                     providers=["CUDAExecutionProvider" if torch.cuda.is_available() else
                                                                                "CPUExecutionProvider"])
    output_tar = f"{output_tar_prefix}_{os.path.basename(transcript_file)}.tar"
    print(f"load model on gpu_id {gpu_id} with tar {transcript_file}", flush=True)
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

                            audio_name = "/".join(wav_file.split("/")[-4:])
                            match = re.search(r'quora_xttsv2_([a-zA-Z]+_[a-zA-Z]+)/', wav_file)
                            if match:
                                xx_xx = match.group(1)  # 获取匹配的 xx_xx
                                speaker = xx_xx.replace('_', ' ')  # 将 "_" 替换为 " "
                            else:
                                speaker = "cosyvoice_中文女"

                            codes_numpy = extract_speech_token(speech_tokenizer_session, wav_file)

                            sample = {'text': text,
                                      'codec_label.npy': codes_numpy.tobytes(),
                                      'speaker': speaker,
                                      "__key__": audio_name}
                            writer.write(sample)

                            count += 1
                            if count % 100 == 0:
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
        default='/lp/data/sythetic_audio/quora/quora_xttsv2_meta_splits/group_1',
        help='Directory containing JSONL transcript files.'
    )

    parser.add_argument(
        '--output_tar_prefix',
        type=str,
        default='/gpfs/public/mmodal/users/liupeng/webdataset/quora_xttsv2/quora_xttsv2',
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
