import matplotlib.pyplot as plt
import numpy as np
import matplotlib

words = "Write a quiz about bits that includes the word elephant".split(" ")
base_model = "yi_6b"
base_array = np.load(f"{base_model}.npy")
model = "yi6b_up9b_m24_16"
array = np.load(f"{model}.npy")
# Transpose the array
base_transposed_array = base_array.T
transposed_array = array.T

base_mean_per_token = base_transposed_array[:, 1:-1].mean(-1)
base_all_mean = base_mean_per_token[1:].mean()
print(f"base mean_per_token: {base_mean_per_token}")
print(f"base all mean: {base_all_mean}")

mean_per_token = transposed_array[:, 1:-1].mean(-1)
all_mean = mean_per_token[1:].mean()
print(f"mean_per_token: {mean_per_token}")
print(f"all mean: {all_mean}")

idx = 5
base_cur_array = base_transposed_array[:idx]
cur_array = transposed_array[:idx]

# Plotting all rows of the transposed array on the same plot with different colors
# plt.figure(figsize=(8, cur_array.shape[1]))
fig, axs = plt.subplots(2, 1, figsize=(8, 48))  # 2 rows, 1 column
# Creating a color map for different lines
# colors = plt.cm.viridis(np.linspace(0, 1, 30))
colors = ['#0077BE',  # Cerulean Blue
          '#DC143C',  # Crimson Red
          '#50C878',  # Emerald Green
          '#FFD700',  # Golden Yellow
          '#DA70D6',  # Orchid Purple
          '#FF5733',  # Persimmon Orange
          '#1E90FF']  # Dodger Blue

for i in range(len(base_cur_array)):
    axs[0].plot(base_cur_array[i], marker='o', color=colors[i], label=f'{words[i+0]}')
    axs[0].legend()
    axs[0].set_title('Yi-6B', fontsize=16)
    axs[0].set_xlim(0, 48)
    axs[0].set_ylabel('cosine', fontsize=14)
    axs[0].grid()

for i in range(len(cur_array)):
    axs[1].plot(cur_array[i], marker='o', color=colors[i], label=f'{words[i+0]}')
    axs[1].legend()
    axs[1].set_title('Yi-9B Initialization', fontsize=16)
    axs[1].set_xlim(0, 48)
    axs[1].set_ylabel('cosine', fontsize=14)
    axs[1].grid()

# plt.title(f'{model}')
plt.xlabel('layer_number', fontsize=14)
# plt.tight_layout()
plt.show()

