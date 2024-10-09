# Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set seaborn style for better aesthetics
sns.set(style="whitegrid")

# Define the directory containing the CSV files
data_dir = ""  # Replace with your actual directory

# List of normalization methods
normalizations = ["log", "min-max", "yeo-johnson", "z-normalization"]

# Initialize an empty list to store DataFrames
dfs = []

# Read and combine primary results CSV files
for norm in tqdm(normalizations):
    file_path = os.path.join(data_dir, f"primary_results_{norm}.csv")
    df = pd.read_csv(file_path)
    df["Normalization"] = norm  # Add a column for normalization method
    dfs.append(df)

# Combine all DataFrames into one
combined_df = pd.concat(dfs, ignore_index=True)

# Keep only the folds (exclude the mean row)
combined_df = combined_df[combined_df["Fold"] != "mean"]

# Convert Year to integer
combined_df["Year"] = combined_df["Year"].astype(int)

# Convert MSE, R2, MAPE to numeric (if not already)
combined_df["MSE"] = pd.to_numeric(combined_df["MSE"], errors='coerce')
combined_df["R2"] = pd.to_numeric(combined_df["R2"], errors='coerce')
combined_df["MAPE"] = pd.to_numeric(combined_df["MAPE"], errors='coerce')

# Question 1: Does the R2 and MAPE vary wildly inside the folds of one year?
# We will compute the standard deviation of R2 and MAPE across folds for each year and normalization.

# Compute standard deviation of R2 and MAPE across folds
std_df = combined_df.groupby(["Normalization", "Year"]).agg({
    "R2": ["mean", "std"],
    "MAPE": ["mean", "std"]
}).reset_index()

# Flatten the MultiIndex columns
std_df.columns = ['Normalization', 'Year', 'R2_mean', 'R2_std', 'MAPE_mean', 'MAPE_std']

# Plot R2_std and MAPE_std over years for each normalization
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# R2 standard deviation plot
sns.lineplot(data=std_df, x="Year", y="R2_std", hue="Normalization", marker="o", ax=axes[0])
axes[0].set_title("Standard Deviation of R² Across Folds Over Years")
axes[0].set_ylabel("R² Standard Deviation")

# MAPE standard deviation plot
sns.lineplot(data=std_df, x="Year", y="MAPE_std", hue="Normalization", marker="o", ax=axes[1])
axes[1].set_title("Standard Deviation of MAPE Across Folds Over Years")
axes[1].set_ylabel("MAPE Standard Deviation")

plt.tight_layout()
plt.show()

# Question 2: What's the correlation between R2 and MAPE for everything, for means?
# We will compute the correlation between mean R2 and mean MAPE across years and normalizations.

# Compute correlation between R2_mean and MAPE_mean for each normalization
corr_df = std_df.groupby("Normalization").apply(lambda x: x["R2_mean"].corr(x["MAPE_mean"])).reset_index()
corr_df.columns = ["Normalization", "Correlation"]

print("Correlation between mean R² and mean MAPE for each normalization:")
print(corr_df)

# Plot the correlations
plt.figure(figsize=(8, 6))
sns.barplot(data=corr_df, x="Normalization", y="Correlation")
plt.title("Correlation between mean R² and mean MAPE for each Normalization")
plt.ylabel("Correlation Coefficient")
plt.show()

# Question 3: How does the R2 go down over years?
# Line plot of R2_mean over years for each normalization

plt.figure(figsize=(12, 6))
sns.lineplot(data=std_df, x="Year", y="R2_mean", hue="Normalization", marker="o")
plt.title("Mean R² Over Years for Each Normalization")
plt.ylabel("Mean R²")
plt.show()

# Question 4: How does the MAPE go down over years?
# Line plot of MAPE_mean over years for each normalization

plt.figure(figsize=(12, 6))
sns.lineplot(data=std_df, x="Year", y="MAPE_mean", hue="Normalization", marker="o")
plt.title("Mean MAPE Over Years for Each Normalization")
plt.ylabel("Mean MAPE")
plt.show()


# 2. Investigate the stability of coefficients
# For example, examine how the coefficients for 'INPBELI_PAST' vary across normalizations and years

coeff_cols = [col for col in combined_df.columns if col.startswith("Coeff_")]

# Melt the DataFrame to long format for plotting
coeff_df = combined_df.melt(id_vars=["Normalization", "Year", "Fold"], value_vars=coeff_cols, var_name="Coefficient", value_name="Value")

# Plot coefficient values over years for each normalization
plt.figure(figsize=(12, 6))
sns.lineplot(data=coeff_df[coeff_df["Coefficient"] == "Coeff_INPBELI_PAST"], x="Year", y="Value", hue="Normalization", marker="o")
plt.title("Coefficient of INPBELI_PAST Over Years for Each Normalization")
plt.ylabel("Coefficient Value")
plt.show()


# 4. Examine whether any normalization consistently outperforms others
mean_performance = std_df.groupby("Normalization").agg({
    "R2_mean": "mean",
    "MAPE_mean": "mean"
}).reset_index()

print("Mean Performance Across Normalizations:")
print(mean_performance)