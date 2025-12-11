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
    'C:/Users/Ebrima/Documents/GitHub/G5_Data-Mining-Project-Repo/Crime_Data_LA_2024.xlsx'
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
# 13. CHI-SQUARE TEST: CRIME SERIOUSNESS × AREA
#============================================================
if "part_1_2" in data.columns:
    contingency = pd.crosstab(data['area_name'], data['part_1_2'])
    chi2, p, dof, expected = chi2_contingency(contingency)

    print("\nChi-Squared Results:")
    print(f"Chi2 = {chi2}, p = {p}, dof = {dof}")

#%%[markdown]
#============================================================
# 14. MODELING: CRIME INTENSITY EXPLAINED BY AREA FEATURES
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
Though negative binomial perform better, both models highlight the complexity of crime dynamics and the need for more comprehensive data to improve predictions.'''
#%%[markdown]
#============================================================
# 15. HARSHITH'S QUESTION — CRIME TRENDS BY MONTH
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
# 16. MUHANNAD’S QUESTION — TEMPORAL PATTERNS
#============================================================

df = data.copy()
df['day_of_week'] = df['date_occ'].dt.day_name()
df['day_num'] = df['date_occ'].dt.dayofweek
df['is_weekend'] = df['day_num'].isin([5,6]).astype(int)

df['hour'] = (df['time_occ'].astype(float) // 100).astype("Int64")

def time_category(h):
    if pd.isna(h): return "Unknown"
    if 0 <= h <= 5: return "Night"
    if 6 <= h <= 11: return "Morning"
    if 12 <= h <= 17: return "Afternoon"
    return "Evening"

df['time_of_day'] = df['hour'].apply(time_category)

df['month'] = df['date_occ'].dt.month
df['month_name'] = df['date_occ'].dt.month_name()

def season(m):
    if m in [12,1,2]: return "Winter"
    if m in [3,4,5]: return "Spring"
    if m in [6,7,8]: return "Summer"
    return "Fall"

df['season'] = df['month'].apply(season)

holiday_dates = pd.to_datetime([
    "2024-01-01","2024-01-15","2024-02-19","2024-05-27",
    "2024-07-04","2024-09-02","2024-10-14",
    "2024-11-11","2024-11-28","2024-12-25"
])

df['is_holiday'] = df['date_occ'].dt.normalize().isin(holiday_dates).astype(int)

# Plot: Crime by Day of Week
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='day_of_week',
              order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
plt.tight_layout()
plt.show()

# Plot: Crime by Time of Day
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='time_of_day',
              order=['Night','Morning','Afternoon','Evening','Unknown'])
plt.tight_layout()
plt.show()

# Plot: Month Patterns
plt.figure(figsize=(10,5))
sns.countplot(
    data=df, 
    x='month_name',
    order=['January','February','March','April','May','June','July','August',
           'September','October','November','December']
)
plt.tight_layout()
plt.show()

#%%[markdown]
from scipy import stats

weekday_total = df[df["is_weekend"] == 0].shape[0]
weekend_total = df[df["is_weekend"] == 1].shape[0]

# Count weekday/weekend days in the year
days_2024 = pd.date_range("2024-01-01", "2024-12-31")
num_weekdays = sum(days_2024.weekday < 5)
num_weekend_days = sum(days_2024.weekday >= 5)
weekday_avg = weekday_total / num_weekdays
weekend_avg = weekend_total / num_weekend_days

# Daily samples
daily_counts = df.groupby(df["date_occ"].dt.date).size()
weekday_sample = daily_counts[pd.to_datetime(daily_counts.index).weekday < 5]
weekend_sample = daily_counts[pd.to_datetime(daily_counts.index).weekday >= 5]

# Welch's t-test
t_stat, p_value = stats.ttest_ind(
    weekday_sample,
    weekend_sample,
    equal_var=False
)

print("Weekday vs Weekend Welch t-test:")
print("t-statistic:", t_stat)
print("p-value:", p_value)


# %%[markdown]
#Interpretation:

'''Hypotheses:
H0: Mean daily crime is the same on weekdays and weekends.
H1: Mean daily crime is different.

Result:
• t ≈ -0.7854  (small)
• p ≈ 0.4328  (very large)

Decision (α = 0.05):
• p > 0.05 → fail to reject H0.
'''
#Conclusion:

'''There is no statistically significant difference in daily crime levels
between weekdays and weekends in Los Angeles during 2024.

Even though weekday totals are higher, the *per-day* crime rate is nearly
identical — and the statistical test confirms there is no meaningful difference.
'''
#============================================================
print("\nALL ANALYSIS COMPLETE.")

# %%
#############################################################
# %%
#END OF PROJECT WORK 
#############################################################
# ============================================================
# THANK YOU FOR REVIEWING OUR CODE!



