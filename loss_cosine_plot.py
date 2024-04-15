import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import numpy as np

# Convert the string data to a pandas DataFrame
data = pd.read_csv("34B_cosine_all_code.csv", header=None)
data.columns = ['Index', 'Iteration', 'Tokens', 'Cosine']


# Function to model (second degree polynomial for smooth curve fitting)
def model_func(x, a, b, c):
    return a * x**2 + b * x + c

# Filter data for tokens greater than 500
filtered_data = data[data['Tokens'] > 700]

# Fit the model to the data
popt, _ = curve_fit(model_func, filtered_data['Tokens'], filtered_data['Cosine'])
print(popt)

# Plotting the original data
plt.figure(figsize=(14, 7))
plt.plot(data['Tokens'], data['Cosine'], marker='o', label='Original Data')

# Plot the smooth curve
tokens_range = np.linspace(800, max(filtered_data['Tokens'] + 1000), 100)
smooth_curve = model_func(tokens_range, *popt)
plt.plot(tokens_range, smooth_curve, 'r-', label='Smooth Curve Fit')

# plt.plot([1000, 1100, 1200], [0.9978257, 0.99763155, 0.9972974], marker='x', color='darkorange', label='Unscheduled')


# Add title and labels
plt.title('Code Yi-34B Cosine Cumulative vs Tokens with Smooth Curve Fit')
# plt.title('English Yi-6B Cosine Cumulative vs Tokens with Smooth Curve Fit')
plt.xlabel('Tokens(B)')
plt.ylabel('Cosine Cumulative')
plt.legend()
plt.grid(True)
plt.show()

