print("TRAIN SCRIPT STARTED")
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Ensure model folder exists
os.makedirs("model", exist_ok=True)

# Load dataset
data = pd.read_csv("data/ai4i2020.csv")

# One-hot encode categorical columns
data = pd.get_dummies(data, drop_first=True)

# Define features and target
X = data.drop("Machine failure", axis=1)
y = data["Machine failure"]

# ✅ SAVE FEATURE NAMES (THIS FIXES THE 500 ERROR)
joblib.dump(X.columns.tolist(), "model/features.pkl")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print("Model Accuracy:", accuracy)

# Save trained model
joblib.dump(model, "model/model.pkl")

print("Model and features saved successfully ✅")