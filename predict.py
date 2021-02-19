# Import libraries
import numpy as np
import pandas as pd
import pipeline.predict_pipeline as pp

state = input('State: ')
property_valuation = input('Property Valuation: ')
gender = input('Gender')
owns_car = input('Own a car (Yes/No): ')
past_3_years_bike_related_purchases = input('past_3_years_bike_related_purchases: ')
job_industry_category = input('job_industry_category: ')
wealth_segment = input('Wealth segment: ')
tenure = int(input('Tenure: '))

features = pd.DataFrame({'state':state, 'property_valuation':property_valuation, 'owns_car':owns_car, \
                                'past_3_years_bike_related_purchases':past_3_years_bike_related_purchases, \
                                 'job_industry_category':job_industry_category, 'wealth_segment':wealth_segment, \
                                 'tenure':tenure, 'gender':gender}, index=[0])

transformed_values = pp.predict(features)