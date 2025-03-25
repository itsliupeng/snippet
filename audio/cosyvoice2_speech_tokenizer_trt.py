import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically initializes CUDA driver
import whisper


class SpeechTokenizerTRT:
    def __init__(self, engine_path):
        # 初始化 TensorRT Logger
        self.logger = trt.Logger(trt.Logger.VERBOSE)

        # 反序列化引擎
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        # 创建执行上下文
        self.context = self.engine.create_execution_context()
        self.cfx = cuda.Device(0).make_context()

        # 分配输入和输出缓冲区
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

    def allocate_buffers(self):
        # 根据引擎的绑定信息分配缓冲区
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            is_input = self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT
            binding_shape = self.engine.get_tensor_shape(binding)
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))

            print(f"Name='{binding}', Mode={'Input' if is_input else 'Output'}, Shape={binding_shape}, DType={dtype}")

            # 排除批量维度（通常是第0维）
            # 如果网络使用了显式批量维度，确保输入数据包含批量维度
            size = trt.volume(binding_shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                inputs.append({'host': host_mem, 'device': device_mem, 'shape': binding_shape})
            else:
                outputs.append({'host': host_mem, 'device': device_mem, 'shape': binding_shape})

        return inputs, outputs, bindings, stream

    def extract_speech_token(self, file):
        """
        根据音频文件提取 speech token。
        """
        self.cfx.push()

        # 处理音频文件
        speech = whisper.load_audio(file, sr=16000)
        speech = speech[..., : 16000 * 30]
        token_len = int(speech.shape[-1] / 16000 * 25)
        speech = torch.from_numpy(speech).cuda()
        speech = whisper.pad_or_trim(speech).unsqueeze(0)
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        # feat = feat.half()

        # 假设模型输入需要的 shape 和类型与原先一致
        feat_np = feat.detach().cpu().numpy().ravel()  # 展平成一维数组
        lengths_np = np.array([feat.shape[2]], dtype=np.int32).ravel()


        # **Copy data into pre-allocated host buffers**
        np.copyto(self.inputs[0]['host'], feat_np)
        np.copyto(self.inputs[1]['host'], lengths_np)

        print(f"Input 0 (feats) - Shape: {self.inputs[0]['shape']}, Data Type: {self.inputs[0]['host'].dtype}")
        print(f"Input 1 (feats_length) - Shape: {self.inputs[1]['shape']}, Data Type: {self.inputs[1]['host'].dtype}")

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

        self.cfx.pop()
        return speech_token[:token_len]


# 使用示例
if __name__ == "__main__":
    # cuda.init()
    print(f"Number of CUDA Devices: {cuda.Device.count()}")
    for i in range(cuda.Device.count()):
        device = cuda.Device(i)
        print(f"Device {i}: {device.name()}, Memory: {device.total_memory() // (1024 ** 2)} MB")

    # 初始化 SpeechTokenizerTRT
    engine_path = "/lp/models/CosyVoice2-0.5B/speech_tokenizer_v2_fp16.plan"  # TensorRT engine 文件路径
    tokenizer = SpeechTokenizerTRT(engine_path)

    # 示例音频文件路径
    # file = "/lp/data/sythetic_audio/quora/quora_xttsv2/quora_xttsv2_Ana_Florence/part_27/part-00859-1202a768-344d-4f07-8c2c-de78455d4be2-c000/line_00003_0000.wav"
    file = "/lp/data/sythetic_audio/quora/quora_xttsv2/quora_xttsv2_Ana_Florence/part_27/part-00859-1202a768-344d-4f07-8c2c-de78455d4be2-c000/line_00006_0000.wav"

    # 提取 speech token
    token = tokenizer.extract_speech_token(file)

    print("Extracted Speech Token:", token)
