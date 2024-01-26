import os
import sys
import networkx as nx
import torch.multiprocessing as mp


# 重新定义之前的函数
def min_deg(G):
    """返回图G中度数最小的节点"""
    return min(G, key=G.degree)


def neighbors(G, node, visited):
    """返回节点的未访问邻居"""
    return [n for n in G.neighbors(node) if not visited[n]]


def dfs_with_weight(G):
    """基于权重的深度优先遍历"""
    P = []
    visited = {node: False for node in G.nodes()}

    while not all(visited.values()):
        # 选择度数最小的节点
        current_node = min_deg(G)
        pp = [current_node]
        # print(f"new start {current_node}")
        visited[current_node] = True

        while True:
            # 获取未访问的邻居
            unvisited_neighbors = neighbors(G, current_node, visited)
            if not unvisited_neighbors:
                break

            # 选择权重最大的邻居
            max_weighted_neighbor = max(
                unvisited_neighbors,
                key=lambda n: G[current_node][n].get('weight', 0)
            )
            pp.append(max_weighted_neighbor)
            visited[max_weighted_neighbor] = True
            current_node = max_weighted_neighbor

        # 移除已访问的节点
        G.remove_nodes_from([node for node, v in visited.items() if v])
        P.append(pp)

    return P


def producer(queue, file_list):
    for filename in file_list:
        print(f'load file {filename}', flush=True)
        prev_seed = None
        for line in open(filename):
            src_id, dst_id, score, seed = line.strip().split(",")
            src_id, dst_id, score, seed = int(src_id), int(dst_id), float(score), int(seed)

            if prev_seed is None or prev_seed != seed:
                # new graph
                # if prev_seed is not None:
                #     print(G)
                G = nx.Graph(name=str(seed))
                # G_list.append(G)
                queue.put(G)
                prev_seed = seed

            G.add_edge(src_id, dst_id, weight=score)
        # end file
        queue.put(G)


def consumer(queue, split_idx, idx):
    dst_dir = LINK_DATA_DIR
    of = open(f"{dst_dir}/link_path_{split_idx}_{idx}.txt", 'w')
    idx = 0
    while True:
        G = queue.get()
        idx += 1
        if idx % 1000 == 0:
            print(f"processing {idx} graphs")

        if G is not None:
            all_path_list = dfs_with_weight(G)
            for path_list in all_path_list:
                of.write(",".join(map(str, path_list)))
                of.write("\n")
            of.flush()
        else:
            print('G is None, exiting...')
            of.close()
            exit(1)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


if __name__ == '__main__':
    NUM_CONSUMER = 96
    NUM_PRODUCER = 20
    DATA_DIR = sys.argv[1]
    split_idx = int(sys.argv[2])

    LINK_DATA_DIR = DATA_DIR + "_separate_link"
    os.makedirs(LINK_DATA_DIR, exist_ok=True)

    print(f'processing DATA_DIR {DATA_DIR}')
    G_queue = mp.Queue(maxsize=4096)

    saver_list = []
    for i in range(NUM_CONSUMER):
        saver_proc = mp.Process(target=consumer, args=(G_queue, split_idx, i))
        saver_proc.start()
        saver_list.append(saver_proc)
        print(f"Starting saver {i}...")

    file_list = os.listdir(DATA_DIR)
    file_list = list(map(lambda y: os.path.join(DATA_DIR, y), filter(lambda x: x.endswith(".csv"), file_list)))
    file_list = sorted(file_list)

    file_list = file_list[120 * split_idx: 120 * (split_idx+1)]
    print(f"all file_list: {file_list}")
    # 按照文件大小升序

    split_file_list = []
    for pp in chunks(file_list, int(len(file_list) / NUM_PRODUCER)):
        split_file_list.append(pp)

    producer_list = []
    for i in range(NUM_PRODUCER):
        print(f"Starting loader {i}...")
        producer_proc = mp.Process(target=producer, args=(G_queue, split_file_list[i]))
        producer_proc.start()
        producer_list.append(producer_proc)

    print("Waiting for producer to complete...")
    [p.join() for p in producer_list]

    # all end
    for _ in range(NUM_CONSUMER):
        G_queue.put(None)

    print("Waiting for saver to complete...")
    [s.join() for s in saver_list]

    print('Done')
