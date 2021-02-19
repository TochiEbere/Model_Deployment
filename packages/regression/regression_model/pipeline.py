import numpy as np
import pandas as pd
from regression_model.preprocessing import preprocessors as proc
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso


PIPELINE_NAME = 'pipeline'

profit_pipeline = Pipeline([('drop_features', proc.DropUnecessaryFeatures()),
                            ('encoder', proc.EncodeData()),
                             ('fillNA', proc.FillMissingValues()),
                            ('lasso_model', Lasso(random_state=0))
                             ])