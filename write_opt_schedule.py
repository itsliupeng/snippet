import os
import torch

dst_dir = "/lp/model/33b_iter_0488000/iter_0488000"
dir_list = os.listdir(dst_dir)
from_path = "/ML-A100/home/niuxinyao/VonNeumann/llama_ds_z1_nl60_hs7168_gb960_mb1/big_cow_brother_33B/checkpoint/iter_0488000/mp_rank_00_000/model_optim_rng.pt"
m0 =torch.load (from_path, map_location=torch.device('cpu'))
print(m0['opt_param_scheduler'])

for cur_dir in dir_list:
    cur_dir = os.path.join(dst_dir, cur_dir)
    filename = f"{cur_dir}/model_optim_rng.pt"
    cur_m = torch.load(filename, map_location=torch.device('cpu'))
    cur_m['opt_param_scheduler'] = m0['opt_param_scheduler']
    print(f"saving file {filename}")
    torch.save(cur_m, filename)