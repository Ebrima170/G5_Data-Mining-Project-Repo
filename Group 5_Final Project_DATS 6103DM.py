#%%[markdown]
#Team 5: Harshith, Muhannad, Ebrima
#Group 5 Project Work: 
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

data = pd.read_excel('C:/Users/Ebrima/Documents/GitHub/G5_Data-Mining-Project-Repo/Crime_Data_LA_2024.xlsx')
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
