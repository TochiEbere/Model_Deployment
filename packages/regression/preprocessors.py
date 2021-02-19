# Import libraries
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class EncodeData(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['gender'].replace({'M':'Male', 'Femal':'Female', 'F':'Female', '247':'Female', 'U':'Female'}, inplace=True)
        X['gender'].replace({'Female' : 1, 'Male' : 2}, inplace=True)
        X['job_industry_category'].replace({'Entertainment' : 1, 'Telecommunications' : 2, 'IT' : 3,\
                                               'Manufacturing' : 4, 'Financial Services' : 5, 'Retail' :\
                                               6, 'Health' : 7, 'Property' : 8, 'Argiculture' : 9}, inplace=True)
        X['state'].replace({'Victoria' : 'VIC', 'New South Wales' : 'NSW'}, inplace=True)
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
            for feat in features:
                X[feat].fillna(fill_with_vals[feat], inplace=True)
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
        scaler  = StandardScaler()
        X = scaler.fit_transform(X)

        return X
        
class DropUnecessaryFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X[['state', 'property_valuation', 'gender', 'owns_car', 'past_3_years_bike_related_purchases', 'job_industry_category',\
                  'wealth_segment', 'tenure']]
        return X 