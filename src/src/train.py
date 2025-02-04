import os
import argparse
import numpy as np
import pandas as pd
import joblib
from google.cloud import storage, aiplatform
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--bucket", type=str, required=True, help="GCS bucket to save the model")
parser.add_argument("--project", type=str, required=True, help="Google Cloud Project ID")
parser.add_argument("--region", type=str, required=True, help="GCP region for Vertex AI")
args = parser.parse_args()

# Ensure proper formatting of the bucket name
BUCKET_NAME = args.bucket.replace("gs://", "").strip("/")  # Remove "gs://" prefix if present
PROJECT_ID = args.project
REGION = args.region

# Generate synthetic fraud detection data
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=5,
    n_redundant=2,
    n_clusters_per_class=1,
    weights=[0.95, 0.05],
    random_state=42
)

# Convert to Pandas DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
df['label'] = y

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df.drop("label", axis=1), df["label"], test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(solver="liblinear")
model.fit(X_train, y_train)

# Save the trained model locally first
local_model_path = "models/fraud_model.pkl"
os.makedirs("models", exist_ok=True)
joblib.dump(model, local_model_path)

print(f"Model training complete. Saved locally to {local_model_path}")

# **Upload the trained model to GCS**
def upload_to_gcs(local_path, bucket_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)  # Use only the bucket name (no gs://)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to gs://{bucket_name}/{destination_blob_name}")

# Set the Cloud Storage path properly
gcs_model_path = f"models/fraud_model.pkl"  # Just the path inside the bucket

# Upload to GCS
upload_to_gcs(local_model_path, BUCKET_NAME, gcs_model_path)

# **Register Model in Vertex AI Model Registry**
aiplatform.init(project=PROJECT_ID, location=REGION)

model = aiplatform.Model.upload(
    display_name="fraud-detection-model",
    artifact_uri=f"gs://{BUCKET_NAME}/models",  # Use only bucket + directory
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
)

print(f"Model registered successfully in Vertex AI: {model.resource_name}")