import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("SampleSuperstore.csv")

# Basic Information
print("Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Descriptive Statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Mean, Median, Count
print("\nMean Sales:", df["Sales"].mean())
print("Median Sales:", df["Sales"].median())
print("Count Sales:", df["Sales"].count())

print("\nMean Profit:", df["Profit"].mean())
print("Median Profit:", df["Profit"].median())

# Histogram
plt.figure(figsize=(8,5))
plt.hist(df["Sales"], bins=20)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()

# Boxplot
plt.figure(figsize=(8,5))
sns.boxplot(x=df["Sales"])
plt.title("Sales Outlier Detection")
plt.show()

# Correlation Heatmap
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# Top Categories by Sales
category_sales = df.groupby("Category")["Sales"].sum()

plt.figure(figsize=(8,5))
category_sales.plot(kind="bar")
plt.title("Sales by Category")
plt.ylabel("Total Sales")
plt.show()

print("\nEDA Completed Successfully")