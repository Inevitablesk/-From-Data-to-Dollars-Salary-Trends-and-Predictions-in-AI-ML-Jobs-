import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the dataset
df = pd.read_csv("salaries.csv")

# --- Step 1: Data Cleaning ---
print("\n--- Dataset Info ---")
print(df.info())

# Check and handle missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Drop rows with missing salary or job title
df = df.dropna(subset=['salary_in_usd', 'job_title'])

# Remove outliers using IQR
Q1 = df['salary_in_usd'].quantile(0.25)
Q3 = df['salary_in_usd'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['salary_in_usd'] >= Q1 - 1.5 * IQR) & (df['salary_in_usd'] <= Q3 + 1.5 * IQR)]

# Filter for AI/ML/Data roles
keywords = ["data", "ml", "machine", "ai", "artificial", "scientist", "engineer"]
df_filtered = df[df['job_title'].str.lower().str.contains('|'.join(keywords))]

# Drop unnecessary columns
if 'salary_currency' in df_filtered.columns:
    df_filtered = df_filtered.drop(columns=['salary_currency'])

# --- Basic Details Using describe() ---
print("\n--- Describe() Summary for salary_in_usd ---")
print(df_filtered['salary_in_usd'].describe())

# --- Step 2: Basic Statistics ---
print("\n--- Basic Statistics on Salaries (USD) ---")
print("Mean Salary: ", df_filtered['salary_in_usd'].mean())
print("Median Salary: ", df_filtered['salary_in_usd'].median())
print("Mode Salary: ", df_filtered['salary_in_usd'].mode()[0])
print("Standard Deviation: ", df_filtered['salary_in_usd'].std())
print("Minimum Salary: ", df_filtered['salary_in_usd'].min())
print("Maximum Salary: ", df_filtered['salary_in_usd'].max())

def manual_label_encoding(column):
    unique_vals = column.unique()
    mapping = {val: idx for idx, val in enumerate(unique_vals)}
    return column.map(mapping), mapping

encoded_df = df_filtered.copy()
categorical_columns = ['experience_level', 'employment_type', 'job_title',
                       'employee_residence', 'company_location', 'company_size']

encoders = {}
for col in categorical_columns:
    encoded_df[col], encoders[col] = manual_label_encoding(encoded_df[col])
    # --- Step 3: Label Encoding ---
def manual_label_encoding(column):
    unique_vals = column.unique()
    mapping = {val: idx for idx, val in enumerate(unique_vals)}
    return column.map(mapping), mapping

encoded_df = df_filtered.copy()
categorical_columns = ['experience_level', 'employment_type', 'job_title',
                       'employee_residence', 'company_location', 'company_size']

encoders = {}
for col in categorical_columns:
    encoded_df[col], encoders[col] = manual_label_encoding(encoded_df[col])

