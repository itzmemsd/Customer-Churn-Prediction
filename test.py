import pandas as pd

# Load dataset
df = pd.read_csv("dataset/Telco-Customer-Churn.csv")

# Display first five records
print(df.head())

# Display dataset information
print("\nDataset Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())