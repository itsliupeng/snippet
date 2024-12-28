import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 自动初始化 CUDA 驱动
import whisper


class SpeechTokenizerTRT:
    def __init__(self, engine_path):
        # 初始化 TensorRT Logger
        self.logger = trt.Logger(trt.Logger.WARNING)

        # 反序列化引擎
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        # 创建执行上下文
        self.context = self.engine.create_execution_context()

        # 分配输入和输出缓冲区
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

    # def allocate_buffers(self):
    #     # 根据引擎的绑定信息分配缓冲区
    #     inputs = []
    #     outputs = []
    #     bindings = []
    #     stream = cuda.Stream()
    #
    #     for binding in self.engine:
    #         binding_idx = self.engine.get_binding_index(binding)
    #         binding_shape = self.engine.get_binding_shape(binding)
    #         dtype = trt.nptype(self.engine.get_binding_dtype(binding))
    #         size = trt.volume(binding_shape) * self.engine.max_batch_size
    #         host_mem = cuda.pagelocked_empty(size, dtype)
    #         device_mem = cuda.mem_alloc(host_mem.nbytes)
    #         bindings.append(int(device_mem))
    #         if self.engine.binding_is_input(binding):
    #             inputs.append({'host': host_mem, 'device': device_mem})
    #         else:
    #             outputs.append({'host': host_mem, 'device': device_mem})
    #
    #     return inputs, outputs, bindings, stream

    def allocate_buffers(self):
        # 根据引擎的绑定信息分配缓冲区
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            binding_name = self.engine.get_binding_name(binding)
            binding_shape = self.engine.get_tensor_shape(binding)
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))

            # 排除批量维度（通常是第0维）
            # 如果网络使用了显式批量维度，确保输入数据包含批量维度
            size = trt.volume(binding_shape) * self.engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem, 'shape': binding_shape})
            else:
                outputs.append({'host': host_mem, 'device': device_mem, 'shape': binding_shape})

        return inputs, outputs, bindings, stream

    def extract_speech_token(self, file):
        """
        根据音频文件提取 speech token。
        """
        # 处理音频文件
        speech = whisper.load_audio(file, sr=16000)
        speech = np.expand_dims(speech, 0)  # [1, audio_length]
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)  # [1, 128, time]

        # 假设模型输入需要的 shape 和类型与原先一致
        feat_np = feat.numpy().astype(np.float32).ravel()  # 展平成一维数组
        lengths_np = np.array([feat.shape[2]], dtype=np.int32).ravel()

        # 准备输入数据
        # 假设输入绑定索引为 0 和 1
        # 请根据实际引擎的输入绑定顺序调整索引
        self.inputs[0]['host'] = feat_np
        self.inputs[1]['host'] = lengths_np

        # 将输入数据复制到设备
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        cuda.memcpy_htod_async(self.inputs[1]['device'], self.inputs[1]['host'], self.stream)

        # 执行推理
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # 从设备复制输出数据到主机
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output['host'], output['device'], self.stream)

        # 同步流
        self.stream.synchronize()

        # 假设输出绑定索引为 0
        # 请根据实际引擎的输出绑定顺序调整索引
        speech_token = self.outputs[0]['host']

        # 根据模型输出的实际形状重塑数组
        # 假设输出 shape 为 [1, token_length, token_dim]
        # 需要根据实际情况调整
        speech_token = speech_token.reshape(1, -1, speech_token.shape[-1])

        return speech_token[0]


# 使用示例
if __name__ == "__main__":
    # 初始化 SpeechTokenizerTRT
    engine_path = "/lp/models/CosyVoice2-0.5B/model.plan.fp32"  # TensorRT engine 文件路径
    tokenizer = SpeechTokenizerTRT(engine_path)

    # 示例音频文件路径
    audio_file = "/path/to/audio.wav"

    # 提取 speech token
    token = tokenizer.extract_speech_token(audio_file)

    print("Extracted Speech Token:", token)
