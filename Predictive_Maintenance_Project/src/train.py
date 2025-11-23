
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from huggingface_hub import HfApi, login, ModelCard, ModelCardData, hf_hub_download
import numpy as np

# Hugging Face Config
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USER = os.environ.get("HF_USER", "RajendrakumarPachaiappan")

# Model with best Parameter value
BEST_MODEL_NAME = "Random Forest"
MODEL_PARAMS = {'n_estimators': 200, 'max_depth': 10, 'random_state': 42}

# Hugging Face Datast and Model repo
DATASET_REPO_ID = f"{HF_USER}/engine-predictive-data"
REPO_ID_MODEL = f"{HF_USER}/engine-predictive-model"

MODEL_FILENAME = f'final_{BEST_MODEL_NAME.lower().replace(" ", "_")}_model.joblib'
SCALER_FILENAME = 'standard_scaler.joblib'
TARGET_COL = 'Engine_Condition'

# Dataset path
TRAIN_FILE_PATH = 'data/train_data_scaled.csv'
TEST_FILE_PATH = 'data/test_data_scaled.csv'

# Feature columns
FEATURE_COLS = [
    'Engine_RPM', 'Lub_Oil_Pressure', 'Fuel_Pressure',
    'Coolant_Pressure', 'Lub_Oil_Temperature', 'Coolant_Temperature'
]

def run_pipeline():

    print("Starting Automated ML Pipeline...")

    # Authenticate with Hugging Face
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable not set. Cannot proceed with upload.")
        return

    login(token=HF_TOKEN)
    api = HfApi(token=HF_TOKEN)


    try:
        print(f"Downloading scaled train data from: {DATASET_REPO_ID}/{TRAIN_FILE_PATH}")
        train_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename=TRAIN_FILE_PATH, repo_type="dataset")
        df_train = pd.read_csv(train_path)

        print(f"Downloading scaled test data from: {DATASET_REPO_ID}/{TEST_FILE_PATH}")
        test_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename=TEST_FILE_PATH, repo_type="dataset")
        df_test = pd.read_csv(test_path)

        print("Scaled data files downloaded successfully.")

    except Exception as e:
        print(f"CRITICAL ERROR: Failed to download scaled data files. Please ensure they are at '{TRAIN_FILE_PATH}' and '{TEST_FILE_PATH}' in the dataset repository.")
        print(f"Underlying Error: {e}")
        return

    # Prepare data
    X_train = df_train[FEATURE_COLS]
    y_train = df_train[TARGET_COL]

    X_test = df_test[FEATURE_COLS]
    y_test = df_test[TARGET_COL]

    print(f"Data loaded: Train shape {X_train.shape}, Test shape {X_test.shape}")

    print("Creating and saving identity StandardScaler for deployment consistency")

    n_features = len(FEATURE_COLS)
    identity_scaler = StandardScaler()
    # Manually set attributes for an identity transformation: mean=0, std=1
    identity_scaler.mean_ = np.zeros(n_features)
    identity_scaler.scale_ = np.ones(n_features)
    identity_scaler.n_features_in_ = n_features

    # Save identity scaler locally
    joblib.dump(identity_scaler, SCALER_FILENAME)


    # Model Training
    print(f"Training final {BEST_MODEL_NAME} Model with params: {MODEL_PARAMS}")
    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    final_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    print(f"Model F1-Score on test set: {final_metrics['f1_score']:.4f}")

    # Save Model locally
    joblib.dump(model, MODEL_FILENAME)

    api.create_repo(repo_id=REPO_ID_MODEL, repo_type="model", exist_ok=True)

    # Upload Model
    api.upload_file(path_or_fileobj=MODEL_FILENAME, path_in_repo=MODEL_FILENAME, repo_id=REPO_ID_MODEL)
    print(f"Uploaded Model: {MODEL_FILENAME}")

    # Upload Scaler
    api.upload_file(path_or_fileobj=SCALER_FILENAME, path_in_repo=SCALER_FILENAME, repo_id=REPO_ID_MODEL)
    print(f"Uploaded Identity Scaler: {SCALER_FILENAME}")

    # Update Model Card
    MODEL_PARAMS_CARD = MODEL_PARAMS.copy()
    MODEL_PARAMS_CARD['features'] = FEATURE_COLS

    metrics_for_card = {k: final_metrics[k] for k in final_metrics if k != 'roc_auc'}

    card_data = ModelCardData(
        library_name="scikit-learn",
        pipeline_tag="structured-data-classification",
        metrics=[{"name": key, "type": key, "value": value} for key, value in metrics_for_card.items()],
        parameters=MODEL_PARAMS_CARD
    )
    card = ModelCard.from_template(card_data)
    card.save("README.md")

    api.upload_file(path_or_fileobj="README.md", path_in_repo="README.md", repo_id=REPO_ID_MODEL)
    print("Model Card (README.md) updated.")

    print(f"Pipeline complete. New model artifacts pushed to {REPO_ID_MODEL}")

if __name__ == "__main__":
    run_pipeline()
