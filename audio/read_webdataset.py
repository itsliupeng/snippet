
import webdataset as wds
from snac import SNAC

import os
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ.pop('SLURM_PROCID', None)

# Define the path to the .tar file
dataset_path = "/lp/pretrain_audio_data/webdataset/quora_xttsv2/quora_xttsv2_part_aa.tar"
dataset = wds.WebDataset(dataset_path)
raw = next(iter(dataset))

import torch
import numpy as np
codec_label=torch.from_numpy(np.frombuffer(raw['wav.codec_label.npy'], dtype=np.int64)).view(7, -1)


#########################

class SnacConfig:
    audio_vocab_size = 4096
    padded_vocab_size = 4160
    end_of_audio = 4097

def reconscruct_snac(output_list):
    if len(output_list) == 8:
        output_list = output_list[:-1]
    output = []
    # for i in range(7):
    #     output_list[i] = output_list[i][i + 1 :]
    for i in range(len(output_list[-1])):
        output.append("#")
        for j in range(7):
            cur_token = output_list[j][i]
            if cur_token < SnacConfig.audio_vocab_size:
                output.append(cur_token)
            else:
                print(f"cur_token {cur_token} oob!!!")
                output.append(0)
    return output

def reconstruct_tensors(flattened_output, device=None):
    """Reconstructs the list of tensors from the flattened output."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def count_elements_between_hashes(lst):
        try:
            # Find the index of the first '#'
            first_index = lst.index("#")
            # Find the index of the second '#' after the first
            second_index = lst.index("#", first_index + 1)
            # Count the elements between the two indices
            return second_index - first_index - 1
        except ValueError:
            # Handle the case where there aren't enough '#' symbols
            return "List does not contain two '#' symbols"

    def remove_elements_before_hash(flattened_list):
        try:
            # Find the index of the first '#'
            first_hash_index = flattened_list.index("#")
            # Return the list starting from the first '#'
            return flattened_list[first_hash_index:]
        except ValueError:
            # Handle the case where there is no '#'
            return "List does not contain the symbol '#'"

    def list_to_torch_tensor(tensor1):
        # Convert the list to a torch tensor
        tensor = torch.tensor(tensor1)
        # Reshape the tensor to have size (1, n)
        tensor = tensor.unsqueeze(0)
        return tensor

    flattened_output = remove_elements_before_hash(flattened_output)
    codes = []
    tensor1 = []
    tensor2 = []
    tensor3 = []

    n_tensors = count_elements_between_hashes(flattened_output)
    if n_tensors == 7:
        for i in range(0, len(flattened_output), 8):

            tensor1.append(flattened_output[i + 1])
            tensor2.append(flattened_output[i + 2])
            tensor3.append(flattened_output[i + 3])
            tensor3.append(flattened_output[i + 4])

            tensor2.append(flattened_output[i + 5])
            tensor3.append(flattened_output[i + 6])
            tensor3.append(flattened_output[i + 7])
            codes = [
                list_to_torch_tensor(tensor1).to(device),
                list_to_torch_tensor(tensor2).to(device),
                list_to_torch_tensor(tensor3).to(device),
            ]
    return codes


token_list = codec_label.numpy().tolist()
audiolist = reconscruct_snac(token_list)
audio_codes = reconstruct_tensors(audiolist)
print([x.shape for x in audio_codes])
snacmodel = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()
with torch.inference_mode():
    audio_hat = snacmodel.decode(audio_codes)


with torch.inference_mode():
    audio_hat_A = snacmodel.decode(codes)


import soundfile as sf
sf.write("test_C.wav", audio_hat.squeeze().cpu().numpy(), 24000)
sf.write("test_A.wav", audio_hat_A.squeeze().cpu().numpy(), 24000)