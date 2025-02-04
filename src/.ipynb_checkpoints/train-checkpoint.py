import os
import numpy as np
import pandas as pd
import joblib
import yaml
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load hyperparameters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Generate synthetic fraud detection data (same as Phase 0)
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=5,
    n_redundant=2,
    n_clusters_per_class=1,
    weights=[0.95, 0.05],
    random_state=params["model"]["random_state"]
)

# Convert to Pandas DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
df['label'] = y

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    df.drop("label", axis=1),
    df["label"],
    test_size=params["model"]["test_size"],
    random_state=params["model"]["random_state"]
)

# Train Logistic Regression model
model = LogisticRegression(solver=params["model"]["solver"])
model.fit(X_train, y_train)

# Save model to the appropriate storage path
platform = os.getenv("PLATFORM", "local")  # Default to local training
with open("config/platform_config.yaml", "r") as f:
    config = yaml.safe_load(f)

if platform == "vertex_ai":
    model_path = config["vertex_ai"]["model_path"]
elif platform == "sagemaker":
    model_path = config["sagemaker"]["model_path"]
elif platform == "azure_ml":
    model_path = config["azure_ml"]["model_path"]
else:
    model_path = "models/fraud_model.pkl"

joblib.dump(model, model_path)
print(f"Model training complete. Saved to {model_path}")

# Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))