# ✅ Phase 1: Run the training job "locally" in your notebook
## 1️⃣ Set up environment variables
Before running train.py, make sure to define the necessary environment variables in your terminal: 
  ```sh
  export BUCKET_NAME=phase1gcp
  export PROJECT_ID=swift-setup-383812
  export REGION=us-central1
  ```
Verify:
  ```sh
  echo $BUCKET_NAME  # Should print 'phase1gcp'
  echo $PROJECT_ID   # Should print your actual project ID
  ```
--------
## 2️⃣ Install dependencies (if needed)
 Ensure that you have all required libraries installed in your notebook environment:
 ```sh
 pip install numpy pandas scikit-learn google-cloud-storage google-cloud-aiplatform joblib
 ```
--------
## 3️⃣ Run the training script
Now, run train.py in your terminal. Make sure you're in the right directory:
  ```sh
  python src/train.py --bucket gs://$BUCKET_NAME --project $PROJECT_ID --region $REGION
  ```
What This Does:
* Generates synthetic fraud detection data.
* Trains a Logistic Regression model.
* Saves fraud_model.pkl locally.
* Uploads fraud_model.pkl to GCS (gs://phase1gcp/models/).
* Registers the model in Vertex AI Model Registry.

✅ If this runs successfully, the model will be in your bucket and Vertex AI Model Registry.

# ✅ Phase 2: Deploy the model to a Vertex AI endpoint
## 1️⃣ Set up variables
We're adding a few more environment variables to what we set during phase 1:
  ```sh 
  export BUCKET_NAME=phase1gcp
  export PROJECT_ID=swift-setup-383812
  export REGION=us-central1
  export MODEL_NAME=fraud-detection-model
  export ENDPOINT_NAME=fraud-detection-endpoint
  ```
## 2️⃣ Create an endpoint
First, create an endpoint in Vertext AI where the model will be deployed:
  ```sh
  gcloud ai endpoints create \
  --region=$REGION \
  --display-name=$ENDPOINT_NAME
  ```
This will return an *Endpoint ID*. Save this ID because we'll need it for the next step.

## 3️⃣ Deploy the model to the endpoint
Run the following command after retrieving the ENDPOINT_ID:
  ```sh
  export ENDPOINT_ID=your-endpoint-id  # Replace with actual ID from previous step
  ```

Run the following command to retrieve the MODEL_ID.
  ```sh
  gcloud ai models list --region=$REGION
  ```
It will return something like below. Copy the ID.
```ID                 DISPLAY NAME             CREATE TIME
123456789012345678 fraud-detection-model    2025-02-04T20:30:12```

Store the MODEL_ID as an environment variable:
  ```sh
  export MODEL_ID=your-model-id  # Replace with actual ID
  ```

Now deploy the model:
  ```sh
  gcloud ai endpoints deploy-model $ENDPOINT_ID \
  --region=$REGION \
  --model=$MODEL_ID \
  --machine-type=n1-standard-4 \
  --min-replica-count=1 \
  --max-replica-count=1 \
  --traffic-split=0=100 \
  --display-name=$ENDPOINT_NAME
  ```
Explanation:
* --machine-type=n1-standard-4: Specifies the VM type for inference.
* --min-replica-count=1: Minimum number of instances.
* --max-replica-count=1: Ensures a single replica to minimize cost.
* --traffic-split=0=100: Sends 100% of traffic to this model version.

## 4️⃣ Test the deployment
Once the deployment is successful, you can send a test inference request. First, retrieve and copy the ENDPOINT_ID (although you should have already stored it as an environment variable):
  ```sh
  gcloud ai endpoints list --region=$REGION
  ```
Then verify the endpoint using `describe`:
  ```sh
  gcloud ai endpoints describe $ENDPOINT_ID --region=$REGION
  ```

Vertex AI requires OAuth 2.0 authentication, so you'll need to generate an access token and include it in your request:
  ```sh
  export ACCESS_TOKEN=$(gcloud auth print-access-token)
  ```
Verify the access token. You should see a long string:
  ```sh
  echo $ACCESS_TOKEN
  ```

Finally, send a test inference request using `curl`:
  ```sh
  curl -X POST \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                   0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
  }' \
  https://us-central1-aiplatform.googleapis.com/v1/projects/$PROJECT_ID/locations/$REGION/endpoints/$ENDPOINT_ID:predict
  ```
If the authentication worked, you should receive a prediction response like:
  ```json
  {
  "predictions": [0]
}
```


  




 