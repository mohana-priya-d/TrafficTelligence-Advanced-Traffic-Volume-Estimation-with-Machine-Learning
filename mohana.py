# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 12:38:08 2025

@author: Mani
"""
# ✅ Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ✅ Load the dataset
df = pd.read_csv(r'C:\Users\Mani\Desktop\TrafficTelligence\traffic volume.csv')

# ✅ View initial data
print("First 5 rows:")
print(df.head())
print("\nInfo:")
print(df.info())
print("\nSummary:")
print(df.describe())

# ✅ Handle missing values
print("\nMissing values before handling:")
print(df.isnull().sum())

df['holiday'] = df['holiday'].fillna('None')
df['weather'] = df['weather'].fillna(df['weather'].mode()[0])
df['temp'] = df['temp'].fillna(df['temp'].mean())
df['rain'] = df['rain'].fillna(0)
df['snow'] = df['snow'].fillna(0)

print("\nMissing values after filling:")
print(df.isnull().sum())

# ✅ Create datetime and extract features
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['Time'], format='%d-%m-%Y %H:%M:%S')
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year

# ✅ Encode categorical values
le = LabelEncoder()
df['holiday'] = le.fit_transform(df['holiday'].astype(str))
df['weather'] = le.fit_transform(df['weather'].astype(str))

# ✅ Drop unnecessary columns
df = df.drop(['date', 'Time', 'datetime', 'month', 'year'], axis=1)

# ✅ Final feature set (7 features)
X = df[['holiday', 'temp', 'rain', 'snow', 'weather', 'hour', 'dayofweek']]
y = df['traffic_volume']

# ✅ Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# ✅ Save the model and scaler
joblib.dump(model, r'C:\Users\Mani\Desktop\TrafficTelligence\traffic_model.pkl')
joblib.dump(scaler, r'C:\Users\Mani\Desktop\TrafficTelligence\scaler.pkl')
print("✅ Model and scaler saved successfully!")

# ✅ Evaluate the model
y_pred = model.predict(X_test)
print("🔮 Sample predictions:")
print(y_pred[:5])
print(f"📊 MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"📈 R² Score: {r2_score(y_test, y_pred):.2f}")