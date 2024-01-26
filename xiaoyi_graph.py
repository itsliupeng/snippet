import copy
import random
import statistics


def create_graph(num, deg):
    graph = dict()
    for v in range(num):
        graph[v] = set()
    for v in range(num):
        for i in random.sample(range(num), deg):
            if i == v: continue
            w = random.random()
            graph[v].add((i,w))
            graph[i].add((v,w))
    print(f'Average number of edges: {statistics.mean([len(n) for v,n in graph.items()])}')
    return graph


def min_degree(graph):
    best = [None, len(graph)+1]  # [node, degree]
    for v,n in graph.items():
        if len(n) < best[1]:
            best = [v, len(n)]
    return best[0]

def dfs_paths(original_graph):
    paths = []
    visited = set()
    graph = copy.deepcopy(original_graph)
    while len(graph) > 0:
        #print(graph)
        path = []
        node = min_degree(graph)
        path.append([node, -1])
        visited.add(node)
        while True:
            neighbors = [n for n in graph[node] if n[0] not in visited]
            if len(neighbors) == 0: break
            neighbors = sorted(neighbors, key=lambda x:x[1], reverse=True)
            node, weight = neighbors[0]  # max weight
            path.append([node, weight])
            visited.add(node)
        paths.append(path)
        # Update graph by removing visited nodes
        for v in visited:
            for nw in graph[v]:
                n, w = nw
                graph[n].remove((v, w))
            graph.pop(v)
        visited = set()
    return paths


class UnionFind:
    def __init__(self):
        self.root_ = dict()
        self.size_ = dict()

    def add(self, x):
        if x in self.root_: return
        self.root_[x] = x
        self.size_[x] = 1

    def root(self, x):
        while x != self.root_[x]:
            self.root_[x] = self.root_[self.root_[x]]
            x = self.root_[x]
        return x

    def find(self, x, y):
        return self.root(x) == self.root(y)

    def union(self, x, y):
        x = self.root(x)
        y = self.root(y)
        if (self.size_[x] < self.size_[y]):
            self.root_[x] = y
            self.size_[y] += self.size_[x]
        else:
            self.root_[y] = x
            self.size_[x] += self.size_[y]


def greedy_paths(graph):
    edges = []  # [(node1, node2, weight)]
    for v,neighbors in graph.items():
        for nw in neighbors:
            n, w = nw
            if v < n:
                edges.append((v, n, w))
    edges = sorted(edges, key=lambda x:x[2], reverse=True)
    graph2 = dict()  # a new graph where degree of each node is <=2
    uf = UnionFind()
    for e in edges:
        n1, n2, w = e
        if n1 in graph2 and len(graph2[n1]) > 1: continue
        if n2 in graph2 and len(graph2[n2]) > 1: continue
        if n1 in graph2 and n2 in graph2 and uf.find(n1, n2): continue
        #print(f'Adding {n1} -- {n2}')
        graph2[n1] = [(n2,w)] if n1 not in graph2 else graph2[n1]+[(n2,w)]
        graph2[n2] = [(n1,w)] if n2 not in graph2 else graph2[n2]+[(n1,w)]
        uf.add(n1)
        uf.add(n2)
        uf.union(n1, n2)
    paths = []
    visited = set()
    for v,ns in graph2.items():
        if len(ns) != 1 or v in visited: continue
        path = [(v, -1)]
        visited.add(v)
        v, w = ns[0]
        while True:
            path.append((v, w))
            visited.add(v)
            ns = [x for x in graph2[v] if x[0] != path[-2][0]]
            if len(ns) == 0: break
            assert len(ns) == 1
            v, w = ns[0]
        paths.append(path)
    for v, ns in graph.items():
        if v not in visited:  # isolated nodes
            paths.append([(v, -1)])
    #print(graph)
    #print(paths)
    return paths


def path_stats(graph, paths):
    lengths = [len(p) for p in paths]
    assert len(graph) == sum(lengths)
    weights = [sum([x[1] for x in p]) for p in paths]
    print(f'Number of nodes: {len(graph)}; number of paths: {len(lengths)}; mean length: {statistics.mean(lengths)}; median length: {statistics.median(lengths)}; max length: {max(lengths)}; weight sum: {sum(weights)}; average weight: {statistics.mean(weights)}')


def dump_graph(graph, outfile):
    edges = []  # [(node1, node2, weight)]
    for v,neighbors in graph.items():
        for nw in neighbors:
            n, w = nw
            edges.append((v, n, w))
    edges = sorted(edges, key=lambda x:x[2], reverse=True)
    with open(outfile, 'w') as o:
        for edge in edges:
            v,n,w = edge
            o.write(f'{v},{n},{w}\n')


if __name__ == '__main__':
    num = 10000
    deg = int(100/2)
    graph = create_graph(num, deg)
    dump_graph(graph, '/mnt/vepfs/home/renxiaoyi/data/cluster/test/simple/test.csv')
    paths = dfs_paths(graph)
    path_stats(graph, paths)
    paths = greedy_paths(graph)
    path_stats(graph, paths)