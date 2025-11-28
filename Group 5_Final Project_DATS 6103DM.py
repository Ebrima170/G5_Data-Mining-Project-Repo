#%%[markdown]
#Team 5: Kadaru, Muhannad, Ebrima
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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# Load dataset
data = pd.read_excel('C:/Users/Ebrima/Documents/GitHub/G5_Data-Mining-Project-Repo/Crime_Data_LA_2024.xlsx')
data = data.iloc[1:].reset_index(drop=True)
# Display first few rows
print(data.head())

# Display dataset info
print(data.info())
# Display summary statistics
print(data.describe())
# Display column names
print(data.columns)
# Display data types for each column
print(data.dtypes)
#print(data.dtypes)
# Check for missing values
print(data.isnull().sum())
'''The following columns have missing values:
Mocodes, Vict Age, Vict Sex, Vict Descent, Premis Cd, Premis Desc, Weapon Used Cd, Weapon Desc, Status, 
Crm Cd 2, Crm cd 3, crm cd 4, Cross Street, LAT, LON
We will handle these missing values in the preprocessing step.
'''
#Vic Sex has X values which is not a valid category, we will treat them as missing values
data['Vict Sex'].replace('X', np.nan, inplace=True)
#Age has 0 values which is not possible, we will treat them as missing values
data['Vict Age'].replace(0, np.nan, inplace=True)

# Preprocessing
# Drop unnecessary columns
data = data.drop(columns=['Mocodes', 'Cross Street', 'Crm Cd 1', 'Crm Cd 2', 
                          'Crm Cd 3', 'Crm Cd 4', 'Weapon Used Cd', 'Weapon Desc', 'AREA'])


# First row contains column descriptions → remove it
data = data.iloc[1:].reset_index(drop=True)

# --------------------------------------------
# 2. Standardize column names
# --------------------------------------------
data.columns = (
    data.columns.str.strip()
              .str.replace(" ", "_")
              .str.replace(".", "", regex=False)
              .str.lower()
)

# --------------------------------------------
# 3. Convert date columns
# --------------------------------------------
data["date_rptd"] = pd.to_datetime(data["date_rptd"]).dt.date
data["date_occ"] = pd.to_datetime(data["date_occ"]).dt.date

# Convert time column
data["time_occ"] = pd.to_datetime(data["time_occ"], format="%H:%M", errors="coerce").dt.time

# --------------------------------------------
# 4. Convert numeric columns
# --------------------------------------------
num_cols = [
    "dr_no", "vict_age", 
    "lat", "lon", "rpt_dist_no"
]

for col in num_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

# --------------------------------------------
# 5. Convert categorical columns
# --------------------------------------------
cat_cols = [
    "area_name", "crm_cd_desc", "vict_sex",
    "vict_descent", "status", "status_desc",
    "premis_desc", "premise_cd", "weapon_desc", 
    "part_1-2", "location", "crm_cd"
]

for col in cat_cols:
    if col in data.columns:
        data[col] = data[col].astype("category")

# --------------------------------------------
# 6. Basic cleaning: remove duplicates
# --------------------------------------------
data = data.drop_duplicates(subset=["dr_no"])

# --------------------------------------------
# 7. Clean unrealistic values
# --------------------------------------------

#----------------------------
#Ensure Victim Sex is either 'M' or 'F' for convenience
#----------------------------

data["vict_sex"] = data["vict_sex"].where(data["vict_sex"].isin(["M", "F"]))

#----------------------------
#Rename part_1-2 variable for clarity
#----------------------------
data = data.rename(columns={"part_1-2": "crime_seriousness"})

# Victim age: valid range 0–120

if "vict_age" in data.columns:
    data["vict_age"] = data["vict_age"].where(data["vict_age"].between(0, 120))

# Latitude/Longitude must be within Los Angeles region
if "lat" in data.columns:
    data["lat"] = data["lat"].where(data["lat"].between(33, 35))

if "lon" in data.columns:
    data["lon"] = data["lon"].where(data["lon"].between(-119, -117))

# --------------------------------------------
# 8. Trim whitespace on all string/categorical columns
# --------------------------------------------
for col in data.select_dtypes(include=["object", "category"]).columns:
    data[col] = data[col].astype(str).str.strip()

# --------------------------------------------
# 9. Output summary of cleaned data
# --------------------------------------------
#print("======== Cleaned Data Summary ========")
#print(data.info())
#print("\nMissing values per column:")
#print(data.isna().sum())

# --------------------------------------------
# 10. Save cleaned dataset
# --------------------------------------------
#data.to_csv("Crime_Data_LA_2024_CLEANED.csv", index=False)

#print("\nSaved cleaned file → Crime_Data_LA_2024_CLEANED.csv")

print(data.dtypes)
