# model_training.py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Step 1: Load the Dataset
df = pd.read_csv("heart.csv")  # Ensure this file is in the same directory

# Step 2: Data Preparation
X = df.drop('target', axis=1)
y = df['target']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Train the Model
model = XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Step 4: Save the Model as a .pkl File
with open("heart_disease_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully as heart_disease_model.pkl")
