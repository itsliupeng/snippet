import torch

for k, v in m['model0']['language_model']['encoder'].items():
    if "_extra_state" in k:
        v.seek(0)
        vv = torch.load(v, "cpu")
        if vv and "amax_history_fwd" in vv and "amax_history_bwd" in vv:
            print(k, vv["amax_history_fwd"].shape, vv["amax_history_bwd"].shape)

import argparse
import os

import torch
import multiprocessing


def check(tpa, tpb):
    m1 = torch.load(f"mp_rank_0{tpa}_000/model_optim_rng.pt", "cpu")
    m2 = torch.load(f"mp_rank_0{tpa}_000/model_optim_rng.pt", "cpu")
    s1 = m1['model0']['language_model']['encoder']["layers.0.self_attention.layernorm_qkv._extra_state"]
    s2 = m2['model0']['language_model']['encoder']["layers.0.self_attention.layernorm_qkv._extra_state"]
    s1.seek(0)
    ss1 = torch.load(s1, "cpu")
    s2.seek(0)
    ss2 = torch.load(s2, "cpu")
    print(ss1['amax_history_fwd'], ss2['amax_history_fwd'])




def check_param_of_part(args):
    part_dir, param = args
    m = torch.load(f"{part_dir}/model_optim_rng.pt", map_location=torch.device("cpu"))
    args_dict = vars(m['args'])
    print(f"in {part_dir}, {param} value is {args_dict[param]}")


def check_params(ckpt_dir, param):
    part_files = sorted(os.listdir(ckpt_dir))

    args_list = list(zip(part_files, [param] * len(part_files)))
    with multiprocessing.Pool(16) as pool:
        pool.map(check_param_of_part, args_list)


check_params(".", "consumed_train_samples")