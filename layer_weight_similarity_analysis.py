import matplotlib.pyplot as plt
import numpy as np

model = "yi_6b_base"
array = np.load(f"{model}.npy")

cur_array = np.array([array[0], array[-1]])

# Plotting all rows of the transposed array on the same plot with different colors
plt.figure(figsize=(cur_array.shape[0], cur_array.shape[1]))

# Creating a color map for different lines
# colors = plt.cm.viridis(np.linspace(0, 1, 5))
colors = ['#0077BE',  # Cerulean Blue
          '#DC143C',  # Crimson Red
          '#50C878',  # Emerald Green
          '#FFD700',  # Golden Yellow
          '#DA70D6']  # Orchid Purple

# words = ['input_norm', 'q', 'k', 'v', 'o', 'gate', 'up', 'down', 'post_norm']
words = ['input_norm', 'post_norm']

for i in range(cur_array.shape[0]):
    plt.plot(cur_array[i], marker='o', color=colors[i], label=f'{words[i]}')

plt.title(f'{model}')
plt.xlabel('layer_number')
plt.ylabel('cosine')
plt.legend()
plt.grid(True)
plt.show()
