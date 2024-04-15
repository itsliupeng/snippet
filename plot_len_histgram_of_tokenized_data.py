import numpy as np
import matplotlib.pyplot as plt

data_dir = "github_repo_humaneval_related_java_content_document.npy"
m_lengths = np.load(data_dir)

# Calculate the 2.5th and 97.5th percentiles to cover 95% of the data
lower_bound = np.percentile(m_lengths, 0.5)
upper_bound = np.percentile(m_lengths, 80)
print(lower_bound, upper_bound)
# lower_bound = 70000
# upper_bound = 300000


# Adjusting bin range to the calculated bounds
bins = np.arange(np.floor(lower_bound), np.ceil(upper_bound), 2000)

# Creating a histogram with the adjusted bins
plt.hist(m_lengths, bins=bins, edgecolor='black')
plt.xlabel('Tokens/Document')
plt.ylabel('Frequency')
plt.title(f'Histogram of {data_dir}')
plt.show()

