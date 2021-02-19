# Import libraries
# from sklearn.base import BaseEstimator, TransformerMixin
# import joblib
# model = joblib.load('Lasso_model.pkl')

import numpy as np
import pandas as pd
import preprocessors as proc
from sklearn.pipeline import Pipeline

CAT_VARS = []

PIPELINE_NAME = 'pipeline'

predict_pipeline = Pipeline([('encoder', proc.EncodeData()), \
                             ('fillNA', proc.FillMissingValues()), ('scaler', proc.ScaleData())])