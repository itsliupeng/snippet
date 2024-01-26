import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Load the data
df = pd.read_csv('ppl.csv')

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of the Variables')
plt.show()

# Replace infinite values with the maximum finite value in the 'intern_7b_ppl' column
max_finite_value = df['intern_7b_ppl'][~np.isinf(df['intern_7b_ppl'])].max()
df['intern_7b_ppl'] = df['intern_7b_ppl'].replace(np.inf, max_finite_value)

# Standardize the data and apply PCA
df_standardized = (df - df.mean()) / df.std()
pca = PCA()
pca.fit(df_standardized)
explained_variance_ratio_after_replacement = pca.explained_variance_ratio_

# Define a function to get the indices of the top and bottom 10% of data
def get_extreme_indices(data):
    lower_threshold = data.quantile(0.10)
    upper_threshold = data.quantile(0.90)
    return data[(data <= lower_threshold) | (data >= upper_threshold)].index

# Get the indices of the top and bottom 10% of data for each column
extreme_indices = {col: get_extreme_indices(df[col]) for col in df.columns}

# Calculate the overlap between each pair of columns
overlap = pd.DataFrame(index=df.columns, columns=df.columns)
for col1 in df.columns:
    for col2 in df.columns:
        overlap.loc[col1, col2] = len(extreme_indices[col1].intersection(extreme_indices[col2]))

# Keep only 'baichuan_13b_ppl', 'llama_7b_ppl' and 'yuchi_ppl'
df_selected = df[['baichuan_13b_ppl', 'llama_7b_ppl', 'yuchi_ppl']]

# Sort the values in each column and divide them into 10 buckets
df_selected_sorted = df_selected.apply(lambda x: pd.qcut(x, 10, labels=False, duplicates='drop'))

# Define a function to count the number of pairs of same buckets
def count_same_pairs(row):
    return sum([row['baichuan_13b_ppl'] == row['llama_7b_ppl'],
                row['baichuan_13b_ppl'] == row['yuchi_ppl'],
                row['llama_7b_ppl'] == row['yuchi_ppl']])

# Apply the function to each row
df_selected_sorted['same_bucket'] = df_selected_sorted.apply(count_same_pairs, axis=1)
df_selected_sorted = df_selected_sorted.rename(columns={'baichuan_13b_ppl': 'baichuan_13b_ppl_bucket',
                                                        'llama_7b_ppl': 'llama_7b_ppl_bucket',
                                                        'yuchi_ppl': 'yuchi_ppl_bucket',
                                                        'same_bucket': 'same_bucket'})

# Concatenate the original dataframe with the bucket indices and the same bucket count
df_final = pd.concat([df, df_selected_sorted], axis=1)

df_final['bucket_mean'] = df_final[['baichuan_13b_ppl_bucket', 'llama_7b_ppl_bucket', 'yuchi_ppl_bucket']].mean(axis=1)
df_final['bucket_std'] = df_final[['baichuan_13b_ppl_bucket', 'llama_7b_ppl_bucket', 'yuchi_ppl_bucket']].std(axis=1)

# Save the final dataframe to a csv file
df_final.to_csv('ppl_buckets.csv')

## select: bucket_mean = [9, 8, 7] and std < 1, != 1
df_bad = df_final.query('bucket_mean >= 7 and bucket_std < 1')
print(f"bad content: {len(df_bad)} / {len(df_final)} = {len(df_bad) / len(df_final)}")


df_bad = df_final.query('bucket_mean >= 9 and bucket_std < 1')
print(f"bad content: {len(df_bad)} / {len(df_final)} = {len(df_bad) / len(df_final)}")


