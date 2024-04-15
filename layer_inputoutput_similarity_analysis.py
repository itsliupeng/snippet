import matplotlib.pyplot as plt
import numpy as np
import matplotlib
# 设置 Matplotlib 的字体
# matplotlib.rcParams['font.family'] = 'serif'  # 例如使用 'SimHei' 字体支持中文
# matplotlib.rcParams['font.serif'] = 'Times New Roman'  # 例如使用 'SimHei' 字体支持中文
# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# plt.rcParams['font.sans-serif']=['Songti']
# plt.rcParams['axes.unicode_minus']=False
#
# words = "<s> Write a qu iz about bits that includes the word ele ph ant".split(" ")
words = "Write a quiz about bits that includes the word elephant".split(" ")
# words = "雷 军 还 拍了 视频 郑 重 让 网友 接受 小米 汽车的 定价 ， 搞 不懂 他的 心态 怎么 是一种 “ 大家 督促 SU 7 卖 低价 ， 成本 高 只能 高价 ” 的 状态 。 SU 7 价格 高 消费者 自然 去 选择 别的 品牌 ， 他想 卖 多少 卖 多少 ， 根本 没人 关心 他的 价格 好不好 。".split(" ")
model = "yi_6b_up9b"
array = np.load(f"{model}.npy")
# Transpose the array
transposed_array = array.T

mean_per_token = transposed_array[:, 1:-1].mean(-1)
all_mean = mean_per_token[1:].mean()
print(f"mean_per_token: {mean_per_token}")
print(f"all mean: {all_mean}")

idx = 5
cur_array = transposed_array[:idx]

# Plotting all rows of the transposed array on the same plot with different colors
plt.figure(figsize=(8, cur_array.shape[1]))

# Creating a color map for different lines
# colors = plt.cm.viridis(np.linspace(0, 1, 30))
colors = ['#0077BE',  # Cerulean Blue
          '#DC143C',  # Crimson Red
          '#50C878',  # Emerald Green
          '#FFD700',  # Golden Yellow
          '#DA70D6',  # Orchid Purple
          '#FF5733',  # Persimmon Orange
          '#1E90FF']  # Dodger Blue

for i in range(len(cur_array)):
    plt.plot(cur_array[i], marker='o', color=colors[i], label=f'{words[i+0]}')
    # plt.ylim(0.2, 1.0)

plt.title(f'{model}')
plt.xlabel('layer_number')
plt.ylabel('cosine')
plt.legend()
plt.grid(True)
plt.show()

####
cur_array = transposed_array[idx:]
# Plotting all rows of the transposed array on the same plot with different colors
plt.figure(figsize=(8, cur_array.shape[1]))

for i in range(len(cur_array)):
    plt.plot(cur_array[i], marker='o', color=colors[i], label=f'{words[i+idx]}')
    plt.ylim(0.2, 1.0)

plt.title(f'{model}')
plt.xlabel('layer_number')
plt.ylabel('cosine')
plt.legend()
plt.grid(True)
plt.show()