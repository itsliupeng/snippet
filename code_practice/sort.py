import os
import random
def quick_sort(l):
    if len(l) <= 1:
        return l
    # pivot = l[len(l) // 2]
    pivot = random.choice(l)
    left = [x for x in l if x < pivot]
    right = [x for x in l if x > pivot]
    middle = [x for x in l if x == pivot]
    return quick_sort(left) + middle + quick_sort(right)


# arr = [3, 6, 8, 10, 1, 2, 1]
# print(quick_sort(arr))  # 输出：[1, 1, 2, 3, 6, 8, 10]


def longestMountain(arr):
    max_length = 0
    for i in range(1, len(arr) -1):
        if arr[i-1] < arr[i] and arr[i] > arr[i+1]:
            # peak
            left, right = i-1, i+1
            while left > 0 and arr[left-1] < arr[left]:
                left -= 1
            while right < len(arr)-1 and arr[right] > arr[right+1]:
                right += 1

            max_length = max(right-left+1, max_length)
    return max_length

# arr =[2,1,4,7,3,2,5]
# print(longestMountain(arr))


def findOrder(numCourses, prerequisites):
    graph = {i: [] for i in range(numCourses)}
    for course, pre in prerequisites:
        graph[pre].append(course)

    visited = set()  # 记录访问过的节点
    res = []

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for neighbor in graph[node]:
            dfs(neighbor)
        res.append(node)  # 后序遍历，先加子节点，再加当前节点

    for i in range(numCourses):
        if i not in visited:
            dfs(i)

    return res[::-1]  # 逆序返回拓扑排序结果

# Test case
numCourses = 6
prerequisites = [[1, 0], [2, 0], [4, 3], [5, 3]]

print(findOrder(numCourses, prerequisites))
