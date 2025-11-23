import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from huggingface_hub import HfApi, login
from datasets import load_dataset

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USER = os.environ.get("HF_USER", "RajendrakumarPachaiappan")
REPO_ID_DATA = f"{HF_USER}/engine-predictive-data"
REPO_ID_MODEL = f"{HF_USER}/engine-predictive-model"

# File Paths
TRAIN_FILE_PATH = 'data/train_data_scaled.csv'
TEST_FILE_PATH = 'data/test_data_scaled.csv'
SCALER_FILENAME = 'standard_scaler.joblib'
DATA_FOLDER = 'data'
os.makedirs(DATA_FOLDER, exist_ok=True)

# Data Constants
FEATURE_COLS = ['Engine_RPM', 'Lub_Oil_Pressure', 'Fuel_Pressure', 'Coolant_Pressure', 'Lub_Oil_Temperature', 'Coolant_Temperature']
TARGET_COL = 'Engine_Condition'
TEST_SIZE = 0.2
RANDOM_STATE = 42
new_columns = {'Engine rpm': 'Engine_RPM', 'Lub oil pressure': 'Lub_Oil_Pressure', 'Fuel pressure': 'Fuel_Pressure', 'Coolant pressure': 'Coolant_Pressure', 'lub oil temp': 'Lub_Oil_Temperature', 'Coolant temp': 'Coolant_Temperature', 'Engine Condition': 'Engine_Condition'}

def run_data_preparation():
    print("Starting Data Preparation Pipeline...")
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set.")
        return

    login(token=HF_TOKEN)
    api = HfApi(token=HF_TOKEN)

    # Load Data
    try:
        print(f"Loading master data from Dataset: {REPO_ID_DATA} (split='master')")
        dataset = load_dataset(REPO_ID_DATA, split='master', token=HF_TOKEN)
        df_master = dataset.to_pandas()
        df_clean = df_master.rename(columns=new_columns)
        print(f"Cleaned data loaded successfully. Shape: {df_clean.shape}")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load data. {e}")
        return

    # Split and Scale Data
    X = df_clean[FEATURE_COLS]
    y = df_clean[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Recombine features/target into the expected CSV format
    X_train_final = pd.DataFrame(X_train_scaled, columns=FEATURE_COLS, index=X_train.index)
    df_train = X_train_final.copy()
    df_train[TARGET_COL] = y_train.values
    
    X_test_final = pd.DataFrame(X_test_scaled, columns=FEATURE_COLS, index=X_test.index)
    df_test = X_test_final.copy()
    df_test[TARGET_COL] = y_test.values

    # Save Locally
    train_file_path_local = os.path.join(DATA_FOLDER, 'train_data_scaled.csv')
    test_file_path_local = os.path.join(DATA_FOLDER, 'test_data_scaled.csv')
    scaler_path_local = os.path.join(DATA_FOLDER, SCALER_FILENAME)
    
    df_train.to_csv(train_file_path_local, index=False)
    df_test.to_csv(test_file_path_local, index=False)
    joblib.dump(scaler, scaler_path_local)

    # Upload to Hugging Face
    api.create_repo(repo_id=REPO_ID_DATA, repo_type='dataset', exist_ok=True)
    api.upload_file(path_or_fileobj=train_file_path_local, path_in_repo=TRAIN_FILE_PATH, repo_id=REPO_ID_DATA, repo_type='dataset')
    api.upload_file(path_or_fileobj=test_file_path_local, path_in_repo=TEST_FILE_PATH, repo_id=REPO_ID_DATA, repo_type='dataset')
    
    api.create_repo(repo_id=REPO_ID_MODEL, repo_type='model', exist_ok=True)
    # Upload scaler to the root of the Model Hub
    api.upload_file(path_or_fileobj=scaler_path_local, path_in_repo=SCALER_FILENAME, repo_id=REPO_ID_MODEL, repo_type='model')

    print("Data preparation complete. Scaled files and StandardScaler pushed to Hugging Face.")

if __name__ == "__main__":
    run_data_preparation()
