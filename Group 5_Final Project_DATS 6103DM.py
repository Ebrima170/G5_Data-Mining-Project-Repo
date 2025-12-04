#%%[markdown]
#Team 5: Harshith, Muhannad, Ebrima
#Group 5 Project Work: ?
'''This project uses a dataset from LPAD on the crime rates in Los Angeles. 
The original data set contains several years, but for the purpose of this project, we are focusing only on the data for 2024.
Description of the variables can be found on the word document.
'''
#%%[markdown]
#Columns in the dataset:
'''['DR_NO', 'Date Rptd', 'DATE OCC', 'TIME OCC', 'AREA', 'AREA NAME',
       'Rpt Dist No', 'Part 1-2', 'Crm Cd', 'Crm Cd Desc', 'Mocodes',
       'Vict Age', 'Vict Sex', 'Vict Descent', 'Premis Cd', 'Premis Desc',
       'Weapon Used Cd', 'Weapon Desc', 'Status', 'Status Desc', 'Crm Cd 1',
       'Crm Cd 2', 'Crm Cd 3', 'Crm Cd 4', 'LOCATION', 'Cross Street', 'LAT',
       'LON']'''

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# Load dataset

data = pd.read_excel('Crime_Data_LA_2024.xlsx')
print("\nRaw data loaded:")
print(data.head())
print(data.info())

# ============================================================
# 2. STANDARDIZE COLUMN NAMES
# ============================================================
data.columns = (
    data.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)
#Drop columns with too many missing values
data = data.drop(columns=['mocodes', 'cross_street', 'crm_cd_1', 'crm_cd_2', 
                          'crm_cd_3', 'crm_cd_4', 'weapon_used_cd', 'weapon_desc'])
# ============================================================
# 3. CLEAN ALL STRING COLUMNS
# Remove spaces, blanks, and placeholders
# ============================================================
for col in data.columns:
    data[col] = (
        data[col]
        .astype(str)                  # convert all to string
        .str.strip()                  # trim spaces
        .replace(
            ["", " ", "nan", "na", "n/a", "none", "unknown", "null"],
            np.nan,
            regex=False
        )
    )

# ============================================================
# 4. FIX DATE COLUMNS
# These columns exist in your dataset
# ============================================================
date_cols = ["date_rptd", "date_occ"]

for col in date_cols:
    if col in data.columns:
        data[col] = pd.to_datetime(data[col], errors="coerce")

#Custom date format parsing if needed
data['date_rptd'] = pd.to_datetime(data['date_rptd'], format='%d/%m/%Y %H:%M', errors='coerce')
data['date_occ']  = pd.to_datetime(data['date_occ'], format='%d/%m/%Y %H:%M', errors='coerce')

# ============================================================
# 5. FIX NUMERIC COLUMNS
# These are the numeric fields you are analyzing
# ============================================================
numeric_cols = ["dr_no", "rpt_dist_no", "vict_age", "lat", "lon"]

for col in numeric_cols:
    if col in data.columns:
        data[col] = (
            data[col]
            .astype(str)
            .str.replace(",", "", regex=False)  # remove commas
        )
        data[col] = pd.to_numeric(data[col], errors="coerce")

# ============================================================
# 6. Clean victim sex field (M/F only, else NaN)
# ============================================================
if "vict_sex" in data.columns:
    data["vict_sex"] = data["vict_sex"].str.upper()
    data["vict_sex"] = data["vict_sex"].replace(
        {"M": "M", "F": "F"},
        regex=False
    )
    data["vict_sex"] = data["vict_sex"].where(
        data["vict_sex"].isin(["M", "F"]), np.nan
    )

# 7. Fix victim age: convert to numeric and clean invalid values
data["vict_age"] = pd.to_numeric(data["vict_age"], errors="coerce")

# Replace impossible/invalid ages with NaN
data.loc[data["vict_age"] == 0, "vict_age"] = np.nan
data.loc[data["vict_age"] < 0, "vict_age"] = np.nan
data.loc[data["vict_age"] > 110, "vict_age"] = np.nan  # cap at plausible human age

#  Drop rows with missing age for plotting
clean_age = data["vict_age"].dropna()
# ============================================================
# 7. Drop rows ONLY where essential numeric data is missing
# ============================================================
required_numeric = ["lat", "lon", "vict_age"]

data = data.dropna(subset=required_numeric)

print("\nDataset AFTER cleaning:")
print(data.head())
print(data.info())
print(data.describe())

# ============================================================
# 8. Diagnostics
# ============================================================
print("\nMissing values per column:")
print(data.isna().sum())

print("\nSummary statistics for numeric columns:")
print(data[numeric_cols].describe())

# ============================================================
# 9. Identify variable types for EDA
# ============================================================
numeric_vars = ["dr_no", "rpt_dist_no", "vict_age", "lat", "lon"]
categorical_vars = ["time_occ", "vict_sex", "premis_cd"]
datetime_vars = ["date_rptd", "date_occ"]

print("\nNumeric variables:", numeric_vars)
print("Categorical variables:", categorical_vars)
print("Datetime variables:", datetime_vars)

# ============================================================
# 10. SAFE plotting (skip empty plots)
# ============================================================
for col in categorical_vars:
    if col not in data.columns:
        print(f"Skipping missing column: {col}")
        continue
    
    plt.figure(figsize=(10, 4))
    data[col].value_counts().head(20).plot(kind="bar")
    plt.title(f"Top 20 values for {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
# ------------------------------------------------------`  `

for col in numeric_vars:
    if col in data.columns:
        if data[col].dropna().empty:
            print(f"\nSkipping histogram for '{col}' (no data).")
            continue
        
        plt.figure(figsize=(8, 4))
        sns.histplot(data[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()  

for col in datetime_vars:
    if col in data.columns:
        if data[col].dropna().empty:
            print(f"\nSkipping time series plot for '{col}' (no data).")
            continue
        
        plt.figure(figsize=(10, 4))
        data.set_index(col).resample('ME').size().plot()
        plt.title(f"Time Series of {col}")
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()  
# %%
# Question 3: Which crime types showed the sharpest increases or decreases in 2024?

import seaborn as sns
import matplotlib.pyplot as plt

data['date_occ'] = pd.to_datetime(data['date_occ'], errors='coerce')
data = data[data['date_occ'].notna()].copy()
data['month'] = data['date_occ'].dt.month

crime_monthly = (
    data.groupby(['crm_cd_desc', 'month'])
        .size()
        .reset_index(name='count')
)

top_crimes = (
    data['crm_cd_desc']
        .value_counts()
        .head(10)
        .index
)

crime_monthly_top = crime_monthly[crime_monthly['crm_cd_desc'].isin(top_crimes)]

plt.figure(figsize=(14, 8))
sns.lineplot(
    data=crime_monthly_top,
    x='month',
    y='count',
    hue='crm_cd_desc',
    marker='o'
)
plt.xticks(range(1, 13))
plt.xlabel('Month')
plt.ylabel('Number of crimes')
plt.title('Monthly Crime Trends for Top 10 Crime Categories in 2024')
plt.legend(title='Crime Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

trend_pivot = crime_monthly.pivot(index='crm_cd_desc', columns='month', values='count').fillna(0)

if 1 in trend_pivot.columns and 12 in trend_pivot.columns:
    trend_pivot['trend'] = trend_pivot[12] - trend_pivot[1]
else:
    first_col = trend_pivot.columns.min()
    last_col = trend_pivot.columns.max()
    trend_pivot['trend'] = trend_pivot[last_col] - trend_pivot[first_col]

rising = trend_pivot.sort_values('trend', ascending=False).head(10)
falling = trend_pivot.sort_values('trend', ascending=True).head(10)

print("Top 10 rising crime types (change from start to end of year):")
print(rising['trend'])

print("\nTop 10 declining crime types (change from start to end of year):")
print(falling['trend'])

# %%
# SIGNIFICANCE TESTING FOR MONTHLY TRENDS (KENDALL'S TAU TEST)

from scipy.stats import kendalltau

trend_tests = []

for crime in crime_monthly['crm_cd_desc'].unique():

    subset = crime_monthly[crime_monthly['crm_cd_desc'] == crime].sort_values('month')

    # Must have at least 3 points to detect a trend
    if subset['month'].nunique() < 3:
        continue

    # Perform Kendall's Tau trend test
    tau, p_val = kendalltau(subset['month'], subset['count'])

    trend_tests.append({
        'crime_type': crime,
        'tau': tau,
        'p_value': p_val,
        'trend_direction': "Increasing" if tau > 0 else "Decreasing",
        'significant': "YES" if p_val < 0.05 else "NO"
    })

trend_tests_df = pd.DataFrame(trend_tests)

significant_increases = (
    trend_tests_df[(trend_tests_df['significant']=="YES") & (trend_tests_df['tau']>0)]
    .sort_values('tau', ascending=False)
    .head(10)
)

significant_decreases = (
    trend_tests_df[(trend_tests_df['significant']=="YES") & (trend_tests_df['tau']<0)]
    .sort_values('tau')
    .head(10)
)

print("\n================ SIGNIFICANT INCREASING TRENDS (Kendall's Tau) ================")
print(significant_increases)

print("\n================ SIGNIFICANT DECREASING TRENDS (Kendall's Tau) ================")
print(significant_decreases)
# %%
