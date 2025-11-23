
from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="Predictive_Maintenance_Project/deployment",

    repo_id="RajendrakumarPachaiappan/EnginePredictionModel",

    repo_type="space",
    path_in_repo="",
)
