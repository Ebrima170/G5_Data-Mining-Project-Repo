#%%[markdown]
#Team 5: Kadaru, Muhannad, Ebrima
#Group 5 Project Work: 

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# Load dataset
data = pd.read_excel('C:/Users/Ebrima/Documents/GitHub/G5_Data-Mining-Project-Repo/Crime_Data_LA_2024.xlsx')
# Display first few rows
print(data.head())

# Display dataset info
print(data.info())
# Display summary statistics
print(data.describe())
# Display column names
print(data.columns)
# Display data types
print(data.dtypes)
# Check for missing values
print(data.isnull().sum())