# ✅ Step 1: Run the training job "locally" in your notebook
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
 ## 2️⃣ Install dependencies (if needeD)
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
 