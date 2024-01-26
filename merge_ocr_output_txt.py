import os
import shutil

dst_dir = 'all'

src_dir = os.listdir(".")
src_dir = list(filter(lambda x: x.startswith("part_"), src_dir))

for d in src_dir:
    sub_dir = os.path.join(d, 'structure')
    for x in os.listdir(sub_dir):
        if not x.endswith(".txt"):
            continue
        src_file = os.path.join(sub_dir, x)
        shutil.copyfile(src_file, os.path.join(dst_dir, x))
