import numpy as np
from sklearn.model_selection import train_test_split


from regression_model import pipeline
from regression_model.preprocessing.data_management import (
    load_dataset, save_pipeline)
from regression_model.config import config


def run_training() -> None:

    # load train data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES], data[config.TARGET],
        test_size=config.TEST_SIZE,
        random_state=0)

    pipeline.profit_pipeline.fit(X_train, y_train)

    # psersist pipeline
    save_pipeline(pipeline_to_persist=pipeline.profit_pipeline)


if __name__=='__main__':
    run_training()