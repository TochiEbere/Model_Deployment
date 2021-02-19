import pathlib
import pandas as pd

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'

TESTING_DATA_FILE = DATASET_DIR / 'test.csv'
TRAINING_DATA_FILE = DATASET_DIR / 'train.csv'
TARGET = 'profit'

FEATURES = []

def save_pipeline(*, pipeline_to_persist) -> None:
    '''Persit the pipeline'''
    model_name = 'lasso.pkl'
    model_path = TRAINED_MODEL_DIR / model_name
    joblib.dump(pipeline_to_persist, model_path)

def run_training() -> None:
    '''Train the model'''
    
    data = pd.read_csv(TRAINING_DATA_FILE)
    
    
    print('Training...')
    
if __name__ == '__main__':
    run_training()