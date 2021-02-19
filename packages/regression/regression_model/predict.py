# Import libraries
import numpy as np
import pandas as pd

from regression_model.preprocessing.data_management import load_pipeline

pipeline_file_name = 'test.csv'
price_pipe = load_pipeline(file_name=pipeline_file_name)

def make_prediction(*, input_data) -> dict:

    data = pd.read_json(input_data)
    prediction = price_pipe.predict(data[config.FEATURES])
    response = {'prediction': prediction}

    return response
