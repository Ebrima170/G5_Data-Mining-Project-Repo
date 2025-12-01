#%%
#############################
## Load & Inspect Data     ##
#############################

# Import libraries
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the Excel file
file_path = "Crime_Data_LA_2024.xlsx"
df = pd.read_excel(file_path)

# Check dataset size
df.shape

# View all column names
df.columns

# Check data types and non-null counts
df.info()

# Display first 5 rows
df.head()

# Random sample of 5 rows
df.sample(5, random_state=1)

# Summary statistics for numeric columns
df.describe()

# Summary statistics for object columns
df.describe(include="object").T

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#
#%%


################
## Clean Data ##
################

# Make a clean copy
df_clean = df.copy()

# Replace empty-like strings ("") with NaN in object columns
obj_cols = df_clean.select_dtypes(include="object").columns
df_clean[obj_cols] = df_clean[obj_cols].apply(
    lambda col: col.str.strip().replace("", np.nan)
)

# Check missing values in key columns
key_cols = ["DATE OCC", "TIME OCC", "AREA", "AREA NAME",
            "Crm Cd", "Crm Cd Desc", "LAT", "LON"]
df_clean[key_cols].isna().sum()

# Inspect min and max TIME OCC values
df_clean["TIME OCC"].min(), df_clean["TIME OCC"].max()

# Check for invalid minute values (>= 60)
invalid_time_mask = (df_clean["TIME OCC"] % 100) >= 60
df_clean[invalid_time_mask][["DATE OCC", "TIME OCC"]].head()

# Check LAT / LON ranges
lat_min = df_clean["LAT"].min()
lat_max = df_clean["LAT"].max()
lon_min = df_clean["LON"].min()
lon_max = df_clean["LON"].max()
lat_min, lat_max, lon_min, lon_max

# Remove invalid geolocation rows (LAT=0 or LON=0)
bad_geo_mask = (df_clean["LAT"] == 0.0) | (df_clean["LON"] == 0.0)
df_clean = df_clean[~bad_geo_mask]


# Count duplicate DR_NO values
df_clean["DR_NO"].duplicated().sum()

# Final structure check
df_clean.info()
df_clean.shape

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#
#%%


##############################
## Feature Engineering      ##
##############################

# Create day-of-week columns
df_clean["DAY_OF_WEEK_NUM"] = df_clean["DATE OCC"].dt.dayofweek
df_clean["DAY_OF_WEEK"] = df_clean["DATE OCC"].dt.day_name()
#df_clean[["DATE OCC", "DAY_OF_WEEK_NUM", "DAY_OF_WEEK"]].head()

# Create weekday/weekend indicator
df_clean["IS_WEEKEND"] = df_clean["DAY_OF_WEEK_NUM"].isin([5, 6]).astype(int)
#df_clean[["DATE OCC", "DAY_OF_WEEK", "IS_WEEKEND"]].head(10)

# Extract hour from TIME OCC
df_clean["HOUR"] = (df_clean["TIME OCC"] // 100).astype(int)
#df_clean[["TIME OCC", "HOUR"]].head(10)

# Create time-of-day category
def time_of_day_from_hour(h):
    if 0 <= h <= 5:
        return "Night"
    elif 6 <= h <= 11:
        return "Morning"
    elif 12 <= h <= 17:
        return "Afternoon"
    else:
        return "Evening"

df_clean["TIME_OF_DAY"] = df_clean["HOUR"].apply(time_of_day_from_hour)
#df_clean[["HOUR", "TIME_OF_DAY"]].head(10)

# Create rush-hour indicator
df_clean["IS_RUSH_HOUR"] = df_clean["HOUR"].isin([7, 8, 9, 16, 17, 18]).astype(int)
#df_clean[["HOUR", "IS_RUSH_HOUR"]].head(10)

# Extract month number and month name
df_clean["MONTH_NUM"] = df_clean["DATE OCC"].dt.month
df_clean["MONTH_NAME"] = df_clean["DATE OCC"].dt.month_name()
#df_clean[["DATE OCC", "MONTH_NUM", "MONTH_NAME"]].head()

# Map month to season
def month_to_season(m):
    if m in [12, 1, 2]:
        return "Winter"
    elif m in [3, 4, 5]:
        return "Spring"
    elif m in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

df_clean["SEASON"] = df_clean["MONTH_NUM"].apply(month_to_season)
#df_clean[["MONTH_NUM", "SEASON"]].head(10)

# Create holiday indicator
holiday_dates = pd.to_datetime([
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27",
    "2024-07-04", "2024-09-02", "2024-10-14",
    "2024-11-11", "2024-11-28", "2024-12-25"
]).normalize()

date_norm = df_clean["DATE OCC"].dt.normalize()
df_clean["IS_HOLIDAY"] = date_norm.isin(holiday_dates).astype(int)
#df_clean[["DATE OCC", "IS_HOLIDAY"]].head(10)

# Preview engineered columns
cols_to_see = [
    "DATE OCC", "TIME OCC", "DAY_OF_WEEK", "IS_WEEKEND",
    "HOUR", "TIME_OF_DAY", "IS_RUSH_HOUR",
    "MONTH_NAME", "SEASON", "IS_HOLIDAY"
]
df_clean[cols_to_see].head(15)

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#
#%%


##########################################
## Exploratory Data Analysis (EDA)      ##
##########################################

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")


###############
## Overview  ##
###############

# Total number of crimes
df_clean.shape[0]


######################################
## Crime Frequency (Top Crime Types)##
######################################

# Table of top 10 crimes
df_clean["Crm Cd Desc"].value_counts().head(10)

# Plot top 10 crimes
top10 = df_clean["Crm Cd Desc"].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top10.values, y=top10.index)
plt.xlabel("Number of Incidents")
plt.ylabel("Crime Type")
plt.title("Top 10 Most Frequent Crime Types in 2024")
plt.tight_layout()
plt.show()


#################################
## Crime by Day of the Week    ##
#################################

order_days = ["Monday", "Tuesday", "Wednesday", "Thursday",
              "Friday", "Saturday", "Sunday"]

# Table
df_clean["DAY_OF_WEEK"].value_counts().reindex(order_days)

# Plot
plt.figure(figsize=(8, 5))
sns.countplot(data=df_clean, x="DAY_OF_WEEK", order=order_days)
plt.xlabel("Day of Week")
plt.ylabel("Number of Incidents")
plt.title("Number of Crimes by Day of Week")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


##################################
## Weekday vs Weekend Analysis  ##
##################################

# Table
df_clean["IS_WEEKEND"].value_counts()

# Plot
weekday_weekend = df_clean["IS_WEEKEND"].map({0: "Weekday", 1: "Weekend"})
plt.figure(figsize=(6, 4))
sns.countplot(x=weekday_weekend)
plt.xlabel("Day Type")
plt.ylabel("Number of Incidents")
plt.title("Crimes on Weekdays vs Weekends")
plt.tight_layout()
plt.show()


##################################
## Crime by Hour of the Day     ##
##################################

# Table
df_clean["HOUR"].value_counts().sort_index()

# Plot
plt.figure(figsize=(10, 5))
sns.countplot(data=df_clean, x="HOUR")
plt.xlabel("Hour of Day (0–23)")
plt.ylabel("Number of Incidents")
plt.title("Crimes by Hour of Day")
plt.tight_layout()
plt.show()


##########################################
## Crime by Time-of-Day Category       ##
##########################################

# Table
df_clean["TIME_OF_DAY"].value_counts()

# Plot
order_tod = ["Night", "Morning", "Afternoon", "Evening"]
plt.figure(figsize=(7, 4))
sns.countplot(data=df_clean, x="TIME_OF_DAY", order=order_tod)
plt.xlabel("Time of Day")
plt.ylabel("Number of Incidents")
plt.title("Crimes by Time of Day")
plt.tight_layout()
plt.show()


############################
## Crime by Month         ##
############################

order_months = ["January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"]

# Table
df_clean["MONTH_NAME"].value_counts().reindex(order_months)

# Plot
plt.figure(figsize=(10, 5))
sns.countplot(data=df_clean, x="MONTH_NAME", order=order_months)
plt.xlabel("Month")
plt.ylabel("Number of Incidents")
plt.title("Crimes by Month in 2024")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


###############################
## Crime by Season           ##
###############################

# Table
df_clean["SEASON"].value_counts()

# Plot
order_seasons = ["Winter", "Spring", "Summer", "Fall"]
plt.figure(figsize=(6, 4))
sns.countplot(data=df_clean, x="SEASON", order=order_seasons)
plt.xlabel("Season")
plt.ylabel("Number of Incidents")
plt.title("Crimes by Season in 2024")
plt.tight_layout()
plt.show()


####################################
## Holiday vs Non-Holiday Crimes ##
####################################

# Table
df_clean["IS_HOLIDAY"].value_counts()

# Plot
holiday_labels = df_clean["IS_HOLIDAY"].map({0: "Non-Holiday", 1: "Holiday"})
plt.figure(figsize=(6, 4))
sns.countplot(x=holiday_labels)
plt.xlabel("Day Type")
plt.ylabel("Number of Incidents")
plt.title("Crimes on Holidays vs Non-Holidays")
plt.tight_layout()
plt.show()


################################
## Crime by LAPD Area         ##
################################

# Table
df_clean["AREA NAME"].value_counts().head(10)

# Plot
area_counts = df_clean["AREA NAME"].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=area_counts.values, y=area_counts.index)
plt.xlabel("Number of Incidents")
plt.ylabel("Area Name")
plt.title("Top 10 LAPD Areas by Crime Count (2024)")
plt.tight_layout()
plt.show()

##################################################
#<<<<<<<<<<<<<<<< End of Section >>>>>>>>>>>>>>>>#
# %%
