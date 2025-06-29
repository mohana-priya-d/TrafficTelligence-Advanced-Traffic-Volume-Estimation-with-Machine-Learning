import joblib
# ✅ Task 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ✅ Task 2: Load the dataset
df = pd.read_csv(r'C:\Users\Mani\Desktop\TrafficTelligence\traffic volume.csv')

# ✅ Task 3: Analyze the data
print("First 5 rows:")
print(df.head())
print("\nInfo:")
print(df.info())
print("\nSummary:")
print(df.describe())

# ✅ Task 4: Handle missing values
print("\nMissing values before handling:")
print(df.isnull().sum())
df['holiday'] = df['holiday'].fillna('None')
df['weather'] = df['weather'].fillna(df['weather'].mode()[0])
if df['temp'].isnull().sum() > 0:
    df['temp'] = df['temp'].fillna(df['temp'].mean())
print("\nMissing values after filling:")
print(df.isnull().sum())

# ✅ Task 5: Data Visualization
# Create datetime column
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['Time'], format='%d-%m-%Y %H:%M:%S')

# Extract date features
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year

# Encode categorical columns
le = LabelEncoder()
df['holiday'] = le.fit_transform(df['holiday'].astype(str))
df['weather'] = le.fit_transform(df['weather'].astype(str))

# Drop unused columns
df = df.drop(['date', 'Time'], axis=1)
X = df[['holiday', 'temp', 'rain', 'snow', 'weather', 'hour', 'dayofweek']]

# 📈 Line plot
plt.figure(figsize=(12, 6))
plt.plot(df['datetime'], df['traffic_volume'], color='blue')
plt.title('Traffic Volume Over Time')
plt.xlabel('DateTime')
plt.ylabel('Traffic Volume')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 📊 Boxplot: Day of week
plt.figure(figsize=(10, 6))
sns.boxplot(x='dayofweek', y='traffic_volume', data=df)
plt.title('Traffic Volume by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Traffic Volume')
plt.show()

# 📊 Boxplot: Weather
plt.figure(figsize=(12, 6))
sns.boxplot(x='weather', y='traffic_volume', data=df)
plt.title('Traffic Volume by Weather Condition')
plt.xlabel('Weather')
plt.ylabel('Traffic Volume')
plt.xticks(rotation=45)
plt.show()

# 📈 Pairplot
sns.pairplot(df[['traffic_volume', 'temp', 'rain', 'snow']])
plt.suptitle('Pairplot of Numeric Features', y=1.02)
plt.show()

# ✅ Task 6: Split into dependent & independent variables
X = df.drop(['traffic_volume', 'datetime'], axis=1)
y = df['traffic_volume']

# ✅ Final check for missing values before modeling
print("\nFinal missing value check:")
print(X.isnull().sum())

# ✅ Drop rows with missing values (if any)
df = df.dropna()
X = df.drop(['traffic_volume', 'datetime'], axis=1)
y = df['traffic_volume']

# ✅ Task 7: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Task 8: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ✅ Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# ✅ Save the model
joblib.dump(model, r'C:\Users\Mani\Desktop\TrafficTelligence\traffic_model.pkl')
print("✅ Model saved successfully!")

# ✅ Save the scaler
joblib.dump(scaler, r'C:\Users\Mani\Desktop\TrafficTelligence\scaler.pkl')
print("✅ Scaler saved successfully!")

# ✅ Load the saved model
model = joblib.load(r'C:\Users\Mani\Desktop\TrafficTelligence\traffic_model.pkl')

# ✅ Predict and evaluate
y_pred = model.predict(X_test)
print("🔮 Predictions from loaded model (first 5):")
print(y_pred[:5])

# ✅ Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"📊 MSE: {mse:.2f}, R²: {r2:.2f}")
