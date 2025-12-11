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


#%%[markdown]
#============================================================
# TEAM 5 — Final Python Script
#Ebrima, Harshith, Muhannad
# Data Mining Project
#============================================================

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import chi2_contingency, kendalltau

#%%[markdown] #Ebrima
#============================================================
# 1. LOAD DATA
#============================================================

data = pd.read_excel(
    'Crime_Data_LA_2024.xlsx'
)
print("\nRaw data loaded:")
print(data.head())
print(data.info())


#============================================================
# 2. STANDARDIZE COLUMN NAMES ONCE
#============================================================
data.columns = (
    data.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
)

#============================================================
# 3. DROP COLUMNS WITH TOO MANY MISSING VALUES
#============================================================
drop_cols = [
    'mocodes','cross street','crm cd 1','crm cd 2',
    'crm cd 3','crm cd 4','weapon used cd','weapon desc'
]
existing_drop_cols = [c for c in drop_cols if c in data.columns]
data = data.drop(columns=existing_drop_cols)


#============================================================
# 4. CLEAN STRING COLUMNS PROPERLY
#============================================================
string_placeholders = ["", " ", "nan", "na", "n/a", "none", "unknown", "null"]

for col in data.columns:
    if data[col].dtype == "object":
        data[col] = data[col].astype(str).str.strip()
        data[col] = data[col].replace(string_placeholders, np.nan)


#============================================================
# 5. FIX DATE COLUMNS
#============================================================
date_cols = ["date rptd", "date occ"]
for col in date_cols:
    if col in data.columns:
        data[col] = pd.to_datetime(data[col], errors='coerce')

#============================================================
# 6. FIX NUMERIC COLUMNS
#============================================================
numeric_cols = ["dr_no", "rpt_dist_no", "vict age", "lat", "lon"]

for col in numeric_cols:
    if col in data.columns:
        data[col] = (
            data[col]
            .astype(str)
            .str.replace(",", "", regex=False)
        )
        data[col] = pd.to_numeric(data[col], errors="coerce")


#============================================================
# 7. CLEAN VICTIM SEX (ONLY M/F ALLOWED)
#============================================================
if "vict_sex" in data.columns:
    data["vict_sex"] = data["vict_sex"].str.upper()
    data["vict_sex"] = data["vict_sex"].where(
        data["vict_sex"].isin(["M","F"]), np.nan
    )


#============================================================
# 8. FIX VICTIM AGE (REMOVE INVALID VALUES)
#============================================================
data["vict_age"] = pd.to_numeric(data["vict_age"], errors="coerce")
data.loc[(data["vict_age"] <= 0) | (data["vict_age"] > 110), "vict_age"] = np.nan


#============================================================
# 9. DROP ROWS MISSING KEY NUMERIC VARIABLES
#============================================================
data = data.dropna(subset=["lat", "lon", "vict_age"])

print("\nDataset AFTER CLEANING:")
print(data.info())
print(data.head())


#============================================================
# 10. IDENTIFY VARIABLE TYPES
#============================================================
numeric_vars = ["dr_no", "rpt_dist_no", "vict_age", "lat", "lon"]
categorical_vars = ["time_occ", "vict_sex", "premis_cd"]
datetime_vars = ["date_rptd", "date_occ"]

#%%[markdown]
#============================================================
# 11. SAFE EDA PLOTTING
#============================================================
for col in categorical_vars:
    if col in data.columns:
        plt.figure(figsize=(10, 4))
        data[col].value_counts().head(20).plot(kind="bar")
        plt.title(f"Top 20 values for {col}")
        plt.tight_layout()
        plt.show()

for col in numeric_vars:
    if col in data.columns and not data[col].dropna().empty:
        plt.figure(figsize=(8, 4))
        sns.histplot(data[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.show()

for col in datetime_vars:
    if col in data.columns and not data[col].dropna().empty:
        # Ensure datetime
        data[col] = pd.to_datetime(data[col], errors="coerce")
        ts = data.dropna(subset=[col]).set_index(col)
        if ts.empty:
            print(f"Skipping {col} — no valid datetime entries.")
            continue
        # Resample and plot
        plt.figure(figsize=(10, 4))
        ts.resample("ME").size().plot()
        plt.title(f"Time Series of {col}")
        plt.xlabel("Month")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()


#%%[markdown]
#============================================================
# 12. SMART QUESTION 1 — HIGHEST CRIME AREAS
#============================================================
#============================================================
crime_counts = (
    data.groupby("area_name")
    .size()
    .reset_index(name="crime_count")
    .sort_values("crime_count", ascending=False)
)

# Top 10 areas
top_10_areas = crime_counts.head(10)

# Sort for readability
top_10_sorted = top_10_areas.sort_values('crime_count', ascending=True)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_10_sorted,
    y='area_name',
    x='crime_count',
    palette='viridis'
)

plt.title("Top 10 Areas with Highest Crime Counts", fontsize=14)
plt.xlabel("Crime Count")
plt.ylabel("Area Name")
plt.tight_layout()
plt.show()


#%%[markdown]
#============================================================
# CHI-SQUARE TEST: CRIME SERIOUSNESS × AREA
#============================================================
if "part_1_2" in data.columns:
    contingency = pd.crosstab(data['area_name'], data['part_1_2'])
    chi2, p, dof, expected = chi2_contingency(contingency)

    print("\nChi-Squared Results:")
    print(f"Chi2 = {chi2}, p = {p}, dof = {dof}")

#%%[markdown]
#============================================================
# MODELING: CRIME INTENSITY EXPLAINED BY AREA FEATURES
#============================================================

area_df = (
    data.groupby("area_name")
        .agg(
            crime_count=("dr_no","count"),
            mean_victim_age=("vict_age","mean"),
            pct_male_victims=("vict_sex",lambda x:(x=="M").mean()),
            pct_female_victims=("vict_sex",lambda x:(x=="F").mean()),
            mean_lat=("lat","mean"),
            mean_lon=("lon","mean"),
            unique_crime_types=("crm_cd_desc","nunique"),
            unique_locations=("location","nunique")
        )
        .dropna()
        .reset_index()
)

X = area_df.drop(columns=["crime_count", "area_name"])
y = area_df["crime_count"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

rf = RandomForestRegressor(n_estimators=500, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("\nModel Performance:")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²:", r2_score(y_test, y_pred))

feat_imp = pd.DataFrame({
    "feature": X.columns,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

print("\nFeature Importances:")
print(feat_imp)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp, x="importance", y="feature")
plt.tight_layout()
plt.show()

'''
The Ramdomn Forest model performance indicates that area features moderately explain crime intensity variations.
However, the low R² suggests the model is not highly predictive, likely due to unmeasured factors influencing crime rates.
Below, we try the Negative Binomial Regression model to see if it improves explanatory power.
'''
#%%[markdown]
# NEGATIVE BINOMIAL REGRESSION
import statsmodels.api as sm
import statsmodels.formula.api as smf
area_df_nb = area_df.copy()
area_df_nb = area_df_nb.dropna()
formula = "crime_count ~ mean_victim_age + pct_male_victims + pct_female_victims + mean_lat + mean_lon + unique_crime_types + unique_locations"
nb_model = smf.glm(formula=formula, data=area_df_nb, family=sm.families.NegativeBinomial()).fit()
print("\nNegative Binomial Regression Summary:")
print(nb_model.summary())

#Rate Ratio, interpretable effects, R squared, AIC
print("\nNegative Binomial Regression Rate Ratios:")
rate_ratios = np.exp(nb_model.params)
print(rate_ratios)
print("\nNegative Binomial Regression AIC:", nb_model.aic)
pseudo_r2 = 1 - nb_model.deviance / nb_model.null_deviance
print("Negative Binomial Regression Pseudo R²:", pseudo_r2)

#Plot negative binomial regression results
predicted_counts = nb_model.predict(area_df_nb)
plt.figure(figsize=(10,6))
plt.scatter(area_df_nb['crime_count'], predicted_counts, alpha=0.7) 
plt.plot([area_df_nb['crime_count'].min(), area_df_nb['crime_count'].max()],
         [area_df_nb['crime_count'].min(), area_df_nb['crime_count'].max()],
         color='red', linestyle='--')
plt.xlabel("Actual Crime Counts")
plt.ylabel("Predicted Crime Counts")
plt.title("Negative Binomial Regression: Actual vs Predicted Crime Counts")
plt.tight_layout()
plt.show()

'''
The Negative Binomial Regression model provides interpretable rate ratios for each predictor.
The pseudo R² indicates a modest explanatory power, suggesting that while area features contribute to crime intensity, 
other unmeasured factors likely play significant roles.
Though negative binomial perform better, both models highlight the complexity of crime dynamics and the need for more comprehensive data to improve predictions.
'''
#%%[markdown]
#============================================================
# 13. SMART QUESTION 2 — TEMPORAL PATTERNS
#============================================================
'''
SMART Question:  
Do crime incidents differ significantly between weekdays and weekends in Los Angeles during 2024? 
If differences exist, are they mainly about how much crime happens,  
or when during the day crime occurs?
'''
#%%
df_time = data.copy()

# Keep only crimes from 2024
df_time = df_time[df_time["date_occ"].dt.year == 2024]
print("Rows in 2024 subset:", df_time.shape[0])
#because the date_rptd contains crimes occurred in previous years but reported in 2024

#%%[markdown]
#============================================================
# Clean TIME_OCC properly
#============================================================
'''
We must ensure time_occ is valid HHMM format.
Remove missing times and times with invalid minutes (>= 60).
'''

# Convert to numeric
df_time["time_occ"] = pd.to_numeric(df_time["time_occ"], errors="coerce")

# Remove rows with missing times
df_time = df_time.dropna(subset=["time_occ"])

# Remove invalid minutes (HH**MM** where MM >= 60)
valid_time = (df_time["time_occ"] % 100) < 60
df_time = df_time[valid_time]

# Extract hour (0–23)
df_time["hour"] = (df_time["time_occ"] // 100).astype(int)

print("Rows after cleaning time:", df_time.shape[0])


#%%[markdown]
#============================================================
# FEATURE ENGINEERING — TEMPORAL VARIABLES
#============================================================

# Day-of-week features
df_time["day_num"] = df_time["date_occ"].dt.dayofweek
df_time["day_of_week"] = df_time["date_occ"].dt.day_name()
df_time["is_weekend"] = df_time["day_num"].isin([5,6]).astype(int)

# Time-of-day category
def time_category(h):
    if 0 <= h <= 5: return "Night"
    if 6 <= h <= 11: return "Morning"
    if 12 <= h <= 17: return "Afternoon"
    return "Evening"

df_time["time_of_day"] = df_time["hour"].apply(time_category)




#%%[markdown]
#============================================================
# EDA — CRIME BY DAY OF WEEK
#============================================================
plt.figure(figsize=(8,5))
sns.countplot(
    data=df_time,
    x="day_of_week",
    order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
)
plt.title("Crimes by Day of Week (2024)")
plt.tight_layout()
plt.show()


#%%[markdown]
#============================================================
# STATISTICAL TEST — DAILY CRIME LEVELS (WELCH'S T-TEST)
#============================================================
'''
Does the daily number of crimes differ between weekdays and weekends?
'''

from scipy import stats

# Daily crime counts
daily_counts = df_time.groupby(df_time["date_occ"].dt.date).size()

weekday_daily = daily_counts[pd.to_datetime(daily_counts.index).weekday < 5]
weekend_daily = daily_counts[pd.to_datetime(daily_counts.index).weekday >= 5]

t_stat, p_val = stats.ttest_ind(
    weekday_daily, weekend_daily, equal_var=False
)

print("\nWelch t-test (Daily Crime Levels):")
print("t-statistic:", t_stat)
print("p-value:", p_val)

# Interpretation
'''
Daily crime levels do NOT differ significantly between weekdays and weekends.
'''


#%%[markdown]
#============================================================
# EDA — HOURLY CRIME PATTERNS
#============================================================
hourly_counts = df_time.groupby(["hour","is_weekend"]).size().reset_index(name="count")
hourly_counts["day_type"] = hourly_counts["is_weekend"].map({0:"Weekday",1:"Weekend"})

plt.figure(figsize=(12,6))
sns.lineplot(
    data=hourly_counts,
    x="hour", y="count",
    hue="day_type", marker="o"
)
plt.title("Crime Volume by Hour (Weekday vs Weekend)")
plt.xticks(range(0,24))
plt.tight_layout()
plt.show()


#%%[markdown]
#============================================================
# STATISTICAL TEST — HOURLY CRIME LEVELS (WELCH'S T-TEST)
#============================================================

hourly_daily = df_time.groupby(
    [df_time["date_occ"].dt.date, "hour"]
).size().reset_index(name="count")

weekday_hour = hourly_daily[pd.to_datetime(hourly_daily["date_occ"]).dt.weekday < 5]["count"]
weekend_hour = hourly_daily[pd.to_datetime(hourly_daily["date_occ"]).dt.weekday >= 5]["count"]

t_stat2, p_val2 = stats.ttest_ind(
    weekday_hour, weekend_hour, equal_var=False
)

print("\nWelch t-test (Hourly Crime Levels):")
print("t-statistic:", t_stat2)
print("p-value:", p_val2)

# Interpretation
'''
Hourly averages also do NOT differ significantly between weekdays and weekends.
'''


#%%[markdown]
#============================================================
# EDA — TIME-OF-DAY CATEGORY DISTRIBUTIONS
#============================================================
order_tod = ["Night","Morning","Afternoon","Evening"]

plt.figure(figsize=(10,6))
sns.countplot(
    data=df_time,
    x="time_of_day",
    order=order_tod,
    hue="is_weekend",
    palette={0:"#4c72b0", 1:"#dd8452"}
)
plt.title("Crimes by Time-of-Day Category (2024)")
plt.tight_layout()
plt.show()


#%%[markdown]
#============================================================
# STATISTICAL TEST — TIME-OF-DAY × WEEKEND (CHI-SQUARE)
#============================================================
from scipy.stats import chi2_contingency

contingency = pd.crosstab(df_time["time_of_day"], df_time["is_weekend"])
chi2, p, dof, expected = chi2_contingency(contingency)

print("\nChi-Square Test (Time-of-Day × Weekend):")
print("Chi-square:", chi2)
print("p-value:", p)
print("Degrees of freedom:", dof)

# Interpretation
'''
There IS a statistically significant difference in crime timing across the day
between weekdays and weekends.
'''


#%%[markdown]
#============================================================
# Summary of SMART Question 2
#============================================================

'''
Daily crime levels: NOT significantly different  
Hourly averages: NOT significantly different  
Time-of-day distribution: SIGNIFICANTLY different  

Conclusion:
The difference between weekday and weekend crime is not about volume,
but *when during the day* the crimes occur.
'''
#%%[markdown]
#============================================================
# 14. SMART Question 3 — CRIME TRENDS BY MONTH
#============================================================

data['date_occ'] = pd.to_datetime(data['date_occ'], errors='coerce')
data['month'] = data['date_occ'].dt.month

crime_monthly = (
    data.groupby(['crm_cd_desc','month']).size().reset_index(name='count')
)

top_crimes = (
    data['crm_cd_desc'].value_counts().head(10).index
)

filtered_monthly = crime_monthly[
    crime_monthly['crm_cd_desc'].isin(top_crimes)
]

plt.figure(figsize=(14, 8))
sns.lineplot(
    data=filtered_monthly,
    x='month',
    y='count',
    hue='crm_cd_desc',
    marker='o'
)
plt.xticks(range(1, 13))
plt.tight_layout()
plt.show()

trend_pivot = (
    crime_monthly
    .pivot(index='crm_cd_desc', columns='month', values='count')
    .fillna(0)
)

trend_pivot['trend'] = trend_pivot.max(axis=1) - trend_pivot.min(axis=1)

print("\nTop rising crime types:")
print(trend_pivot['trend'].sort_values(ascending=False).head(10))

#%%[markdown]
# KENDALL TREND TEST
trend_stats = []
for crime in crime_monthly['crm_cd_desc'].unique():
    subset = crime_monthly[crime_monthly['crm_cd_desc']==crime]
    if subset['month'].nunique() < 3:
        continue
    tau, p_val = kendalltau(subset['month'], subset['count'])
    trend_stats.append({
        'crime_type': crime,
        'tau': tau,
        'p_value': p_val,
        'trend': "Increasing" if tau>0 else "Decreasing",
        'significant': p_val < 0.05
    })

trend_df = pd.DataFrame(trend_stats)

print("\nSignificant Increasing Trends:")
print(trend_df[(trend_df['significant']) & (trend_df['tau']>0)]
      .sort_values('tau', ascending=False).head(10))

print("\nSignificant Decreasing Trends:")
print(trend_df[(trend_df['significant']) & (trend_df['tau']<0)]
      .sort_values('tau').head(10))
'''
The Kendall trend test identifies crime types with statistically significant increasing or decreasing trends over the months.
This helps highlight which crimes are becoming more or less prevalent, guiding resource allocation and prevention strategies.
'''
#%%[markdown]
#============================================================
print("\nALL ANALYSIS COMPLETE.")

# %%
#############################################################
# %%
#END OF PROJECT WORK 
#############################################################
# ============================================================
# THANK YOU FOR REVIEWING OUR CODE!



