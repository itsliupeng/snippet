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
import soundfile as sf
import hashlib
import torch
from snac import SNAC
import whisper
import onnxruntime

SAMPLE_RATE = 24000


speech_tokenizer_model = "/lp/models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx"

# AUDIO_DIR = "/lp/pretrain_audio_data/VoiceAssistant-400K/audio"

AUDIO_DIR = "/lp/pretrain_audio_data/VoiceAssistant-400K/audio"

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


def single_job(ort_session, file):
    with torch.no_grad():
        speech = whisper.load_audio(file, sr=16000)
        speech = speech[..., : 16000 * 30]
        speech = torch.from_numpy(speech).cuda().unsqueeze(0)
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        # feat = feat.half()
        speech_token = ort_session.run(None, {ort_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                                            ort_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32)})[0].flatten()
        return speech_token

def process_json_file(json_file, output_tar_prefix, gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    option = onnxruntime.SessionOptions()
    option.intra_op_num_threads = 1
    option.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CUDAExecutionProvider"]
    speech_tokenizer_session = onnxruntime.InferenceSession(
        speech_tokenizer_model,
        sess_options=option,
        providers=providers
    )

    output_tar = f"{output_tar_prefix}_{os.path.basename(json_file).split('.')[-2]}.tar"
    # index_count_map = defaultdict(int)
    with torch.inference_mode():
        with wds.TarWriter(output_tar) as writer:
            count = 0
            for line in open(json_file, 'r', encoding='utf-8'):
                try:
                    item = json.loads(line.strip())

                    question_audio_path = os.path.join(AUDIO_DIR, item['question_audio'])
                    answer_audio_path = os.path.join(AUDIO_DIR, item['answer_audio'])
                    question_tokens = single_job(speech_tokenizer_session, question_audio_path)
                    answer_tokens = single_job(speech_tokenizer_session, answer_audio_path)

                    text = f"{item['question']}_{item['answer']}"
                    md5_hasher = hashlib.md5()
                    # 更新哈希对象，注意需要将字符串编码为字节
                    md5_hasher.update(text.encode('utf-8'))
                    # 获取十六进制的哈希值
                    md5_hash = md5_hasher.hexdigest()

                    sample = {
                        'question': item['question'],
                        'answer': item['answer'],
                        'question_audio': question_tokens.tobytes(),
                        'answer_audio': answer_tokens.tobytes(),  # [7, S]
                        "__key__": md5_hash
                    }

                    writer.write(sample)

                    count += 1
                    if count % 10 == 0:
                        print(f"Processed {count} samples from {json_file} into {output_tar}")
                except Exception as e:
                    print(f"error {e}")

                # index_count_map[item['index']] += 1
def write_webdataset(json_file_list, output_tar_prefix):
    # Get the number of available CPU cores
    # num_workers = min(cpu_count(), len(json_file_list)) // 4
    num_workers = 32
    print(f"num_workers: {num_workers}")

    # Create a pool of workers
    with Pool(num_workers) as pool:
        pool.starmap(process_json_file, [(json_file, output_tar_prefix, idx % 8) for idx, json_file in enumerate(json_file_list)])


# Get list of JSON files in the current directory
json_data_dir = "/lp/pretrain_audio_data/VoiceAssistant-400K/meta/valid_meta_splits"
json_files = list(map(lambda x: os.path.join(json_data_dir, x), filter(lambda x: x.endswith(".jsonl"), os.listdir(json_data_dir))))
json_files = sorted(json_files)

print(f"json_files: {json_files}")

# Create output path and tar file prefix
output_tar_prefix = 'cosyvoice_voice_assistant_200k_en/'

# Call the function to generate WebDataset using multiprocessing
write_webdataset(json_files, output_tar_prefix)
