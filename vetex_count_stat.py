import os
import multiprocessing
file_list = os.listdir(".")
file_list = list(filter(lambda x: x.startswith("link_path"), file_list))

def count_vertices(filename):
    total_count = 0
    for line in open(filename):
        total_count += len(line.strip().split(","))
    return total_count

def count_unqi_vertices(filename):
    uniq_set = set()
    for line in open(filename):
        for x in line.strip().split(","):
            uniq_set.add(x)
    return len(uniq_set)


def count_all_vertices(filename):
    uniq_set = []
    for line in open(filename):
        for x in line.strip().split(","):
            uniq_set.append(x)
    return uniq_set


pool = multiprocessing.Pool(64)

count_list = pool.map(count_vertices, file_list)
print(sum(count_list))




