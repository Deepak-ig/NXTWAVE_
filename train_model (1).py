import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import os

# Load dataset with no header (first row is also data)
csv_path = "data/landmarks.csv"
df = pd.read_csv(csv_path, header=None)

# The last column is the label
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Train classifier
clf = RandomForestClassifier(n_estimators=300, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
acc = clf.score(X_test, y_test)
print(f"Validation Accuracy: {acc:.2%}")

# Save model
os.makedirs("models", exist_ok=True)
dump(clf, "models/gesture_model.joblib")
print("Model saved to models/gesture_model.joblib")
