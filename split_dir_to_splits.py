import os
import random
import math

def build_splits(src_dir, dst_dir, num_splits):
    filenames = os.listdir(src_dir)
    random.shuffle(filenames)

    n = math.ceil(len(filenames) / num_splits)
    splits = []
    for i in range(num_splits):
        splits.append(filenames[i*n: min(len(filenames), (i+1) * n)])
    for i, l in enumerate(splits):
        with open(os.path.join(dst_dir, f"part_{i}.filenames"), "w") as of:
            for x in l:
                of.write(f"{os.path.join(src_dir, x)}\n")
