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
# import torchaudio

speech_tokenizer_model = "/lp/models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx"


def single_job(ort_session, file):
    with torch.no_grad():
        speech = whisper.load_audio(file, sr=16000)
        speech = speech[..., : 16000 * 30]
        speech = torch.from_numpy(speech).cuda().unsqueeze(0)
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        speech_token = ort_session.run(None, {ort_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                                            ort_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32)})[0].flatten()
        return speech_token

class SpeechTokenExtractor:
    def __init__(self, session, batch_size=16, sr=16000, max_duration=30):
        self.session = session
        self.batch_size = batch_size
        self.sr = sr
        self.max_duration = max_duration

    def extract_speech_tokens(self, files):
        tokens_lens = []
        feats = []
        with torch.inference_mode():
            for file in files:
                speech = whisper.load_audio(file, sr=self.sr) # numpy
                speech = speech[:self.sr * 30]
                speech = torch.from_numpy(speech).cuda()
                duration = min(speech.shape[-1] / self.sr, self.max_duration)
                # speech = whisper.pad_or_trim(speech, length=int(self.max_duration * self.sr))
                feat = whisper.log_mel_spectrogram(speech, n_mels=128)  # [128, time]
                feats.append(feat.cpu().numpy())
                tokens_lens.append(math.ceil(duration * 25))

        batch_feat = np.stack(feats, axis=0)  # [batch_size, 128, time]
        batch_lengths = np.array(tokens_lens, dtype=np.int32)  # [batch_size]

        speech_tokens = self.session.run(
            None,
            {
                self.session.get_inputs()[0].name: batch_feat,
                self.session.get_inputs()[1].name: batch_lengths
            }
        )

        # speech_tokens = np.random.randint(0, 1024, (len(feats), 3000))

        # print(f"speech_tokens shape : {speech_tokens.shape}")

        # processed_speech_tokens = [speech_tokens[i, :tokens_lens[i]] for i in range(len(files))]
        return speech_tokens


def process_json_file(transcript_file, output_tar_prefix, gpu_id, batch_size=16):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    option = onnxruntime.SessionOptions()
    option.intra_op_num_threads = 1
    option.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CUDAExecutionProvider"]
    # providers = ["TensorrtExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    # providers = [
    #     (
    #         "TensorrtExecutionProvider",
    #         {
    #             "trt_fp16_enable": True,  # 启用 FP16
    #             "trt_engine_cache_enable": True,  # 启用引擎缓存
    #             "trt_engine_cache_path": "./trt_cache",  # 缓存路径
    #         },
    #     )
    # ] if torch.cuda.is_available() else ["CPUExecutionProvider"]

    speech_tokenizer_session = onnxruntime.InferenceSession(
        speech_tokenizer_model,
        sess_options=option,
        providers=providers
    )

    extractor = SpeechTokenExtractor(speech_tokenizer_session, batch_size=batch_size, sr=16000, max_duration=30)

    output_tar = f"{output_tar_prefix}_{os.path.basename(transcript_file)}.tar"
    print(f"load model on gpu_id {gpu_id} with tar {transcript_file}", flush=True)
    with torch.inference_mode():
        with wds.TarWriter(output_tar) as writer:
            count = 0
            files_batch = []
            texts_batch = []
            audio_names_batch = []
            speakers_batch = []
            try:
                with open(transcript_file, "rb") as file:
                    for byte_line in file:
                        try:
                            line = byte_line.decode('utf-8', errors='strict')
                        except UnicodeDecodeError as e:
                            print(f"Decoding failed: {e}")
                            continue

                        j = line.strip().split("\t")
                        if len(j) != 2 or j[0] == 'text':  # 跳过 header
                            continue

                        text, wav_file = j[0], j[1]

                        audio_name = "/".join(wav_file.split("/")[-4:])
                        match = re.search(r'quora_xttsv2_([a-zA-Z]+_[a-zA-Z]+)/', wav_file)
                        if match:
                            xx_xx = match.group(1)
                            speaker = xx_xx.replace('_', ' ')
                        else:
                            speaker = "cosyvoice_中文女"

                        files_batch.append(wav_file)
                        texts_batch.append(text)
                        audio_names_batch.append(audio_name)
                        speakers_batch.append(speaker)

                        # 当达到批量大小时，进行批量推理
                        if len(files_batch) == extractor.batch_size:
                            # speech_tokens = extractor.extract_speech_tokens(files_batch)
                            speech_tokens = [single_job(speech_tokenizer_session, x) for x in files_batch]

                            # 写入 Tar
                            for i in range(len(files_batch)):
                                sample = {
                                    'text': texts_batch[i],
                                    'codec_label.npy': speech_tokens[i].tobytes(),
                                    'speaker': speakers_batch[i],
                                    "__key__": audio_names_batch[i]
                                }
                                writer.write(sample)
                                count += 1
                                if count % 10 == 0:
                                    print(f"Processed {count * batch_size} samples from {transcript_file} into {output_tar}")

                            # 清空批量列表
                            files_batch = []
                            texts_batch = []
                            audio_names_batch = []
                            speakers_batch = []

                # 处理剩余的样本
                if len(files_batch) > 0:
                    speech_tokens = extractor.extract_speech_tokens(files_batch)

                    # 写入 Tar
                    for i in range(len(files_batch)):
                        sample = {
                            'text': texts_batch[i],
                            'codec_label.npy': speech_tokens[i].tobytes(),
                            'speaker': speakers_batch[i],
                            "__key__": audio_names_batch[i]
                        }
                        writer.write(sample)
                        count += 1
                        if count % 100 == 0:
                            print(f"Processed {count} samples from {transcript_file} into {output_tar}")

            except Exception as e:
                print(f"Exception {e} for file {transcript_file}")

    print(f"Processed a total of {count} samples from {transcript_file} into {output_tar}")


def write_webdataset(json_file_list, output_tar_prefix, num_processes, batch_size=16):
    # Get the number of available CPU cores
    # num_workers = min(cpu_count(), len(json_file_list), 8*2)
    print(f"num_processes: {num_processes}")

    # Create a pool of workers
    with Pool(num_processes) as pool:
        pool.starmap(process_json_file, [(json_file, output_tar_prefix, idx % 8, batch_size) for idx, json_file in enumerate(json_file_list)])


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

    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
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
    batch_size = args.batch_size

    # Get list of JSON files in the current directory
    # transcript_data_dir = "/lp/dataset/audio_data/librilight_processed/jsonl_splits"
    ts_files = list(map(lambda x: os.path.join(transcript_data_dir, x),  os.listdir(transcript_data_dir)))
    ts_files = sorted(ts_files)

    # Create output path and tar file prefix
    # output_tar_prefix = 'webdataset/librilight'

    # Call the function to generate WebDataset using multiprocessing
    write_webdataset(ts_files, output_tar_prefix, num_processes, batch_size)
