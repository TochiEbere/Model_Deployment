from regression_model.predict import make_prediction
from regression_model.preprocessing.data_management import load_dataset
from regression_model.config import config

def test_make_single_prediction():
    test_data = load_dataset(file_name=config.TESTING_DATA_FILE)
    single_test_json = test_data[0:1].to_json(orient='records')

    subject = make_prediction(input_data=single_test_json)

    assert subject is not None