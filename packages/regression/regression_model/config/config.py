import pathlib
import regression_model

PACKAGE_ROOT = pathlib.Path(regression_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'

TESTING_DATA_FILE = 'test.csv'
TRAINING_DATA_FILE = 'train.csv'
TARGET = 'profit'

FEATURES = ['customer_id','address','postcode','state','country',
            'property_valuation','first_name','last_name','gender','past_3_years_bike_related_purchases',
            'DOB','job_title','job_industry_category','wealth_segment','deceased_indicator','owns_car','tenure'
            ]
TEST_SIZE = 0.1
PIPELINE_FILE_NAME = 'regression_model.pkl'