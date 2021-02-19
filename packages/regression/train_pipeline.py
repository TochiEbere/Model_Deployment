import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from pipeline import profit_pipeline


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'

TESTING_DATA_FILE = DATASET_DIR / 'test.csv'
TRAINING_DATA_FILE = DATASET_DIR / 'train.csv'
TARGET = 'profit'

FEATURES = ['customer_id','address','postcode','state','country',
            'property_valuation','first_name','last_name','gender','past_3_years_bike_related_purchases',
            'DOB','job_title','job_industry_category','wealth_segment','deceased_indicator','owns_car','tenure'
            ]


def save_pipeline(*, pipeline_to_persist) -> None:

    save_file_name = 'regression_model.pkl'
    save_path = TRAINED_MODEL_DIR / save_file_name
    joblib.dump(pipeline_to_persist, save_path)

    print('saved pipeline')

def run_training() -> None:

    # load train data
    data = pd.read_csv(TRAINING_DATA_FILE)

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET],
        test_size=0.1,
        random_state=0)

    profit_pipeline.fit(X_train, y_train)

    # psersist pipeline
    save_pipeline(pipeline_to_persist=profit_pipeline)


if __name__=='__main__':
    run_training()