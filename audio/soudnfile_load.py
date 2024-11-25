
# 对齐 whisper.load_audio


import soundfile as sf
import numpy as np
import librosa

def load_audio_with_soundfile(file_path, target_sr=16000):
    """
    使用 soundfile 加载音频文件并进行预处理，以匹配 whisper.load_audio() 的输出。

    参数:
    - file_path: str, 音频文件路径
    - target_sr: int, 目标采样率 (默认 16000 Hz)

    返回:
    - audio_data: np.ndarray, 预处理后的音频数据
    """
    # 1. 读取音频文件，指定 dtype 为 'int16' 以匹配 FFmpeg 的输出
    audio_data, sr = sf.read(file_path, dtype='int16')

    # 2. 如果音频是多通道，转换为单声道
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # 4. 如果采样率不是目标采样率，进行重采样
    if sr != target_sr:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)

    # 3. 将 int16 数据转换为 float32，并归一化到 [-1, 1]
    audio_data = audio_data.astype(np.float32) / 32768.0

    return audio_data

# 使用示例
file_path = 'audio/10214/10108/10214_10108_000003.opus'

import whisper

# 使用 soundfile 加载并预处理音频
audio_sf_processed = load_audio_with_soundfile(file_path)

# 使用 whisper 加载音频
audio_whisper = whisper.load_audio(file_path)

# 对比两者的差异
difference = np.abs(audio_sf_processed - audio_whisper)
print(f"最大差异: {np.max(difference)}")
print(f"均值差异: {np.mean(difference)}")
