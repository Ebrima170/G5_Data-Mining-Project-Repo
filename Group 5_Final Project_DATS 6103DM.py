#%%[markdown]
#Team 5: Harshith, Muhannad, Ebrima

#Topic: Analysis of the Crime Rate Incidents in Los Angeles: Focus on 2024 Data
#Github Repo Link: https://github.com/Ebrima170/G5_Data-Mining-Project-Repo

#%%[markdown]
#1. CONTEXT AND DATASET DESCRIPTION
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
#%%[markdown]
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

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

#%%[markdown]
print("\nDataset AFTER cleaning:")
#remaining column names and first few rows
print("\nColumn names:")
print(data.columns.tolist())
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


#%%[markdown]
# ============================================================
# 10. SAFE plotting (skip empty plots)
# ============================================================
'''
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
     '''  
#EDA is commented out to avoid long plotting times 
'''
EDA plots show the distributions and trends of key variables. 
While some variables have clear patterns, others may require further investigation.
'''
#%%[markdown]
# The End of Data Cleaning and EDA Section
# --------------------------------------------
#SMART QUESTION TO INVESTIGATE FURTHER:
'''
1. Which Areas had the highest crime concentration in 2024?
'''
#Here goes the code to answer the above question:

# 1. Count crimes per area
crime_counts = (
    data['area_name']
    .value_counts()
    .reset_index()
)
crime_counts.columns = ['area_name', 'crime_count']

# 2. Select top 10 areas
top_10_areas = crime_counts.head(10)

# 3. Print results
print("\nTop 10 Areas with Highest Crime Counts in 2024:")
print(top_10_areas)

# 4. Bar plot
plt.figure(figsize=(12, 6))
sns.barplot(
    data=top_10_areas,
    x='area_name',
    y='crime_count',
    hue='area_name',
    palette='viridis',
    legend=False
)
plt.title('Top 10 Areas with Highest Crime Counts in 2024')
plt.xlabel('Area Name')
plt.ylabel('Crime Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Futher Analysis: Statistical Testing and Modeling
# a. Confirmatory statistical tests (Are crime rates significantly different across areas on the basis of crime seriousness?)
from scipy.stats import chi2_contingency
# Create a contingency table
contingency_table = pd.crosstab(data['area_name'], data['part_1_2'])
# Perform Chi-Squared Test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-Squared Test Results:\nChi2 Statistic: {chi2}\nP-Value: {p}\nDegrees of Freedom: {dof}")
if p < 0.05:
    print("Result: Significant difference in crime rates across areas (reject H0).")
else:
    print("Result: No significant difference in crime rates across areas (fail to reject H0).")
'''
The result of the chi-square test shows that there is a significant difference in crime rates across
 different areas in Los Angeles for the year 2024, as indicated by the p-value being less than 0.05.
   This suggests that certain areas experience higher crime rates compared to others, 
   which could be influenced by various socio-economic and environmental factors. 
   Further investigation into these factors could provide insights into targeted crime prevention strategies. 
'''
#%%[markdown]
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


#b. Modeling (What factors explain crime probability or crime intensity?)
# ============================================================
# MODELING: What factors explain crime intensity?
# Predicting crime_count using victim & location characteristics
# ============================================================

# ------------------------------------------------------------
#
# Aggregate crime intensity
area_df = (
    data.groupby("area_name")
        .agg(
            crime_count = ("dr_no", "count"),      # total crimes
            mean_victim_age = ("vict_age", "mean"),
            pct_male_victims = ("vict_sex", lambda x: (x=='M').mean()),
            pct_female_victims = ("vict_sex", lambda x: (x=='F').mean()),
            mean_lat = ("lat", "mean"),
            mean_lon = ("lon", "mean"),
            unique_crime_types = ("crm_cd_desc", "nunique"),
            unique_locations = ("location", "nunique")
        )
        .reset_index()
)

# Drop any NAs that might occur
area_df = area_df.dropna()
print("\nArea-level Data for Modeling:")
print(area_df.head())
# Define X and y
X = area_df.drop(columns=["crime_count", "area_name"])
y = area_df["crime_count"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Fit model
rf = RandomForestRegressor(n_estimators=500, random_state=42)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Model Performance (Area-Level)")
print("----------------------------------------")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.3f}")


importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

print("\nFeature Importances:")
print(importance_df)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x="importance", y="feature")
plt.title("Feature Importances for Area-Level Crime Intensity")
plt.tight_layout()
plt.show()

# ============================================================
#Harshith's Question
# ============================================================
# %%[markdown]
 #3: Which crime types showed the sharpest increases or decreases in 2024?
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
#=============

print(data.head())
print(data.info())



