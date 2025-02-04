#!/bin/bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=fraud-detection-training \
  --python-package-uris=gs://your-gcp-bucket/src.zip \
  --python-module=src.train \
  --args="--bucket=gs://your-gcp-bucket"
