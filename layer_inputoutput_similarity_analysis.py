import matplotlib.pyplot as plt
import numpy as np

# words = "Write a quiz about bits that includes the word elephant".split(" ")
words = "Write a quiz about bits that includes the word elephant".split(" ")
model = "yi_6b_up9b_text_50B"
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
# colors = plt.cm.viridis(np.linspace(0, 1, 5))
colors = ['#0077BE',  # Cerulean Blue
          '#DC143C',  # Crimson Red
          '#50C878',  # Emerald Green
          '#FFD700',  # Golden Yellow
          '#DA70D6',  # Orchid Purple
          '#FF5733',  # Persimmon Orange
          '#1E90FF']  # Dodger Blue

for i in range(len(cur_array)):
    plt.plot(cur_array[i], marker='o', color=colors[i], label=f'{words[i+0]}')
    plt.ylim(0.7, 1.0)

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
    plt.ylim(0.7, 1.0)

plt.title(f'{model}')
plt.xlabel('layer_number')
plt.ylabel('cosine')
plt.legend()
plt.grid(True)
plt.show()