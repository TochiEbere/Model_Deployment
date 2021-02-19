# Import libraries
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import joblib


class EncodeData(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['gender'].replace({'Female' : 1, 'Male' : 2}, inplace=True)
        X['job_industry_category'].replace({'Entertainment' : 1, 'Telecommunications' : 2, 'IT' : 3,\
                                               'Manufacturing' : 4, 'Financial Services' : 5, 'Retail' :\
                                               6, 'Health' : 7, 'Property' : 8, 'Argiculture' : 9}, inplace=True)
        X['state'].replace({'NSW' : 1, 'VIC' : 2, 'QLD' : 3}, inplace=True)
        X['wealth_segment'].replace({'Affluent Customer' : 1, 'Mass Customer' : 2,\
                                        'High Net Worth' : 3}, inplace=True)
        X['owns_car'].replace({'Yes' : 1, 'No' : 2}, inplace=True)
        
        return X


    
class FillMissingValues(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
        
    def fit(self, X,y=None):
        return self
    
    def transform(self, X):
        fill_with_vals = {'state':1, 'property_valuation':7.52, 'gender':1,\
                          'past_3_years_bike_related_purchases':48.81, \
               'wealth_segment': 2, 'owns_car':1, 'tenure':10.68, 'job_industry_category':4}
        features = X.columns
        for feat in features:
            if X[feat].values=='':
                X[feat] = fill_with_vals[feat]
        return X
    
class ScaleData(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
        
    def fit(self, X, y=None):
        
        return self
    
    def transform(self, X):
        scaler = joblib.load('scaler.pickle')
        data = scaler.transform(X)

        return X
        
