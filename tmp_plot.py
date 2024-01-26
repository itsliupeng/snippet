# Combining the code used to generate the final plot:

import matplotlib.pyplot as plt
import pandas as pd

metric_name = "arc-e_fewshot_25"
# Data
data = {
    "Billion": [100, 100, 200, 200, 300, 300,400, 400,
                500, 500,  600, 600,  700, 700,
                800, 800, 850, 850, 900, 900, 950, 950, 1000, 1000],
    "Experiment": ["re_eval", "icl", "re_eval", "icl", "re_eval", "icl", "re_eval", "icl",
                   "re_eval", "icl", "re_eval", "icl", "re_eval", "icl", "re_eval", "icl",
                   "re_eval", "icl", "re_eval", "icl", "re_eval", "icl", "re_eval", "icl"],
    metric_name: [
59.89, 59.09, 63.01, 63.34, 64.52, 65.61, 64.81, 65.57, 66.84, 65.82, 68.01, 67.72, 67.26, 68.22, 66.92, 68.86, 68.22, 67.26, 67.63, 67.42, 67.85, 69.19, 67.68, 68.52,
    ]
}

# Creating DataFrame
df = pd.DataFrame(data)

# Separating data for two experiments
df_re_eval = df[df["Experiment"] == "re_eval"]
df_icl = df[df["Experiment"] == "icl"]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df_re_eval["Billion"], df_re_eval[metric_name], label="base", color="black", marker='o')
plt.plot(df_icl["Billion"], df_icl[metric_name], label="icl", color="red", marker='o')
plt.title(metric_name)
plt.xlabel("B")
plt.ylabel(metric_name)
plt.legend()
plt.grid(True)
plt.show()
