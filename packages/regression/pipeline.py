# Import libraries
# from sklearn.base import BaseEstimator, TransformerMixin
# import joblib
# model = joblib.load('Lasso_model.pkl')

import numpy as np
import pandas as pd
import preprocessors as proc
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso


PIPELINE_NAME = 'pipeline'

profit_pipeline = Pipeline([('drop_features', proc.DropUnecessaryFeatures()),
                            ('encoder', proc.EncodeData()),
                             ('fillNA', proc.FillMissingValues()),
                            ('lasso_model', Lasso(random_state=0))
                             ])