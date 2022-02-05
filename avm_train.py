#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 11:03:44 2022

@author: alex khablov
"""

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

from regions import make_regions_df, augment_by_region
from projections import project_coordinates
from amenities import augment_by_amenities, top_amenities, get_amentities, amen_cols
from augmentation import augment_by_dummies
from prediction_test import prediction_test

#0 read the data

full_df = pd.read_csv('data/scoperty_sample_data.csv')

#1 describe the data
d = full_df.describe(include='all').T

''' Only these predictors could be used for the most part of the dataset
all other predictors should be excluded or constructed
number_of_units
lat
lon
total_area
construction_date**
living_area
property_type*

* categorials
** date
'''
full_df.property_type.value_counts()
#dataset contains houses almost only.
full_df.living_area.describe()
full_df.total_area.describe()

full_df['price_per_sqm'] = full_df['price']/full_df['living_area']

full_df = project_coordinates(full_df)

#2 filter df

def filter_df(full_df):
    filtered_df = full_df[full_df['living_area'].between(50, 500)]
    filtered_df = filtered_df[filtered_df['total_area'].between(100, 1000)]
    filtered_df = filtered_df[filtered_df['price_per_sqm'].between(2000, 5000)]
    filtered_df = filtered_df[filtered_df['property_type'].isin(
        ['single_family_house', 'multi_family_house'])]
    filtered_df = filtered_df[filtered_df['lat']>0]
    filtered_df = filtered_df[filtered_df['lon']>0]
    filtered_df = filtered_df[~pd.isnull(filtered_df['construction_date'])]
    filtered_df = filtered_df[pd.to_datetime(filtered_df['construction_date'],
                               format='%Y-%m-%d', errors='coerce').notnull()]
    filtered_df = filtered_df[
                (filtered_df['construction_date'].str.startswith('18'))|
                (filtered_df['construction_date'].str.startswith('19'))|
                (filtered_df['construction_date'].str.startswith('20'))]
    return filtered_df.copy()

filtered_df = filter_df(full_df)

#3 make square regions

regions_df = make_regions_df(filtered_df)

#4 match regions to dataframe
filtered_df = augment_by_region(filtered_df, regions_df)

#4.5 augment dataset by amenities
amenities = get_amentities()

filtered_df = augment_by_amenities(filtered_df, amenities, top_amenities)


#5 we are ready to prepare the first model

'''the model is very simple
Because we are talking about houses price in big city
There are two more or less equal parts of the price
Land and House itself.
We know nothing about house type and state(material, condition, number of floors)
So the model is very basic
total_price = land_meter_price*total_area/number_of_units + house_meter_price*living_area

Lets do two dummy variable rows using our region_id
One for land and one for house
Multiply it on total and living areas respectively
And calculate simple model
'''
X_train, X_test, y_train, y_test = train_test_split(
    filtered_df[['region_id', 'living_area', 'total_area', 'number_of_units',
                 'property_type', 'construction_date']+amen_cols],
    filtered_df['price'], test_size=0.33, random_state=42)

X_train, predictors = augment_by_dummies(X_train)

reg = ElasticNet(positive=True, alpha=0.02).fit(X_train[predictors], y_train)
feaure_coeffs = dict(zip(predictors, reg.coef_))
print(feaure_coeffs)

print(reg.score(X_train[predictors], y_train))

#6 let's test the model
X_test, predictors_test = augment_by_dummies(X_test, predictors)


def get_model_score(reg, X_test):
    print(reg.score(X_test[predictors], y_test))
    X_test['prediction'] = reg.predict(X_test[predictors])
    X_test['price'] = y_test
    X_test['err'] = X_test['prediction'] - X_test['price']
    X_test['abs_err'] = abs(X_test['err'])
    X_test['err_prcnt'] = X_test['err']/X_test['price']
    X_test['abs_err_prcnt'] = abs(X_test['err_prcnt'])

    dt = X_test[['prediction', 'price', 'err', 'err_prcnt',
                 'abs_err', 'abs_err_prcnt']].describe().T
    return dt


test_metrics = get_model_score(reg, X_test)
print(test_metrics)

#7 save model
pickle.dump(reg, open('model/model.sav', 'wb'))
pickle.dump(predictors, open('model/predictors.sav', 'wb'))
regions_df.to_parquet('model/regions.parquet')

#8 Final test.
'''
Let's use a few generated houses just to ensure
that the whole pipeline is ready to be used
'''
val_df = prediction_test()
