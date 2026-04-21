# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv('/mnt/data/Sleep_health_and_lifestyle_dataset.csv')

# -----------------------------
# 1. Initial Inspection
# -----------------------------
print("Shape:", df.shape)
print("\nData Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# 2. Visualize Missing Data
# -----------------------------
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Data Heatmap")
plt.show()

# -----------------------------
# 3. Handle Missing Values
# -----------------------------

# Separate numerical and categorical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# Impute numerical columns with MEAN
num_imputer = SimpleImputer(strategy='mean')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Impute categorical columns with MOST FREQUENT (MODE)
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# -----------------------------
# 4. Outlier Detection & Treatment (IQR Method)
# -----------------------------
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Cap outliers instead of removing
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

# -----------------------------
# 5. Verify Cleaning
# -----------------------------
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# -----------------------------
# 6. Save Cleaned Dataset
# -----------------------------
df.to_csv('cleaned_sleep_dataset.csv', index=False)

print("\nCleaned dataset saved successfully!")
