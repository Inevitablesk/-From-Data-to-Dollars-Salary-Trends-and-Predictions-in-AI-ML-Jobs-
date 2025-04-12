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
    # --- Step 4: Enhanced Visualizations ---

# 1. Countplot for Experience Level
plt.figure(figsize=(8,5))
sns.countplot(data=df_filtered, x='experience_level', hue='experience_level', palette='Set2', legend=False)
plt.title("Number of Records by Experience Level")
plt.xlabel("Experience Level")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 2. Violinplot: Salary Distribution by Experience
plt.figure(figsize=(10,6))
sns.violinplot(x='experience_level', y='salary_in_usd', hue='experience_level', data=df_filtered, palette='Set3', legend=False)
plt.title("Salary Distribution by Experience Level")
plt.xlabel("Experience Level")
plt.ylabel("Salary (USD)")
plt.tight_layout()
plt.show()

# 3. Stripplot (Fixed from Swarmplot): Salary by Job Title (Top 10)
top_titles_strip = df_filtered['job_title'].value_counts().head(10).index
strip_df = df_filtered[df_filtered['job_title'].isin(top_titles_strip)]

plt.figure(figsize=(16,6))
sns.stripplot(x='job_title', y='salary_in_usd', hue='job_title', data=strip_df, palette='tab10', dodge=False, size=4, legend=False)
plt.title("Salary Distribution for Top 10 Job Titles (Stripplot)")
plt.xlabel("Job Title")
plt.ylabel("Salary (USD)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 4. Boxplot: Salary vs Experience Level
plt.figure(figsize=(10,6))
sns.boxplot(x='experience_level', y='salary_in_usd', data=df_filtered)
plt.title("Salary vs Experience Level")
plt.xlabel("Experience Level")
plt.ylabel("Salary (USD)")
plt.tight_layout()
plt.show()

# 5. Average Salary by Company Size
plt.figure(figsize=(10,6))
sns.barplot(x='company_size', y='salary_in_usd', data=df_filtered, estimator=np.mean)
plt.title("Average Salary by Company Size")
plt.xlabel("Company Size")
plt.ylabel("Average Salary (USD)")
plt.tight_layout()
plt.show()

# 6. Salary Distribution by Top 15 Locations
top_locations = df_filtered['employee_residence'].value_counts().head(15).index
top_location_df = df_filtered[df_filtered['employee_residence'].isin(top_locations)]

plt.figure(figsize=(14,6))
sns.boxplot(x='employee_residence', y='salary_in_usd', data=top_location_df)
plt.title("Salary Distribution by Top 15 Employee Locations")
plt.xlabel("Employee Location")
plt.ylabel("Salary (USD)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 7. Salary by Top 15 Job Titles
top_titles = df_filtered['job_title'].value_counts().head(15).index
top_titles_df = df_filtered[df_filtered['job_title'].isin(top_titles)]

plt.figure(figsize=(16,6))
sns.boxplot(x='job_title', y='salary_in_usd', data=top_titles_df)
plt.title("Salary by Top 15 Job Titles")
plt.xlabel("Job Title")
plt.ylabel("Salary (USD)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 8. Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(encoded_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# --- Step 5: Model Building (Linear Regression without sklearn) ---

# Define features and target
features = ['experience_level', 'employment_type', 'job_title', 'employee_residence',
            'remote_ratio', 'company_location', 'company_size']
X = encoded_df[features].values
y = encoded_df['salary_in_usd'].values

# Manual Train-Test Split
np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))
X_train, X_test = X[indices[:split]], X[indices[split:]]
y_train, y_test = y[indices[:split]], y[indices[split:]]

# Add bias (intercept) column
X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]

# Calculate weights: theta = (XᵀX)^-1 Xᵀy
theta = np.linalg.inv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ y_train

# Predictions
y_pred = X_test_bias @ theta

# Evaluation
r2 = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))
rmse = np.sqrt(np.mean((y_test - y_pred)**2))

print("\n--- Model Evaluation ---")
print("R2 Score:", round(r2, 3))
print("RMSE:", round(rmse, 2))

# --- Step 6: Predict New Sample Salary Manually ---
sample = {
    'experience_level': encoders['experience_level']['SE'],
    'employment_type': encoders['employment_type']['FT'],
    'job_title': encoders['job_title']['Data Scientist'],
    'employee_residence': encoders['employee_residence']['US'],
    'remote_ratio': 100,
    'company_location': encoders['company_location']['US'],
    'company_size': encoders['company_size']['L']
}

sample_values = np.array([1] + [sample[feat] for feat in features])  # include bias term
predicted_salary = sample_values @ theta

print("\n--- Predicted Salary (USD) for Sample ---")
print("Predicted:", round(predicted_salary, 2))


