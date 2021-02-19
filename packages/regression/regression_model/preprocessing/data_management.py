import pandas as pd
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

from regression_model.config import config


def load_dataset(*, file_name: str) -> pd.DataFrame:
    data = pd.read_csv(f'{config.DATASET_DIR}/{file_name}')
    return data

def save_pipeline(*, pipeline_to_persist) -> None:

    save_file_name = config.PIPELINE_FILE_NAME
    save_path = config.TRAINED_MODEL_DIR / save_file_name
    joblib.dump(pipeline_to_persist, save_path)
  
    print('saved pipeline')

def load_pipeline(*, file_name: str) -> Pipeline:
    file_path = config.TRAINED_MODEL_DIR / file_name
    saved_pipeline = joblib.load(file_path)
    return saved_pipeline