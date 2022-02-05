#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 11:03:44 2022

@author: alex khablov
"""

import pandas as pd
import numpy as np
import geopandas

from pyproj import Transformer
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#0 read the data

full_df = pd.read_csv('data/scoperty_sample_data.csv')

#1 describe the data
d = full_df.describe(include='all').T

''' Only these predictors could be used for the most part of the dataset
all other predictors should be excluded or constructed
neighborhood*
zip_code*
street_name*
street_number*
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
full_df['total_area_by_living'] = full_df['total_area']/full_df['living_area']


transformer = Transformer.from_crs('WGS84', 32632, always_xy=True)


def project_coordinates(data: pd.DataFrame) -> pd.DataFrame:
    """Projects coordinates to meters."""
    lon_proj, lat_proj = transformer.transform(data['lon'].to_numpy(),
                                               data['lat'].to_numpy())
    data_ = data.assign(**{
        'lon_proj': lon_proj,
        'lat_proj': lat_proj,
    })
    data_ = geopandas.GeoDataFrame(data_,
                                 geometry=geopandas.points_from_xy(data_['lon_proj'],
                                                                   data_['lat_proj']))
    return data_

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

def make_regions(filtered_df, min_leaf, region = []):
    if region == []:
        region = [min(filtered_df['lat_proj']),
                  min(filtered_df['lon_proj']),
                  max(filtered_df['lat_proj']),
                  max(filtered_df['lon_proj']),]
    if len(filtered_df) >= min_leaf*2:
        if region[2]-region[0] > region[3]-region[1]:
            split = filtered_df['lat_proj'].median()
            region_left = region.copy()
            region_right = region.copy()
            region_left[2] = split
            region_right[0] = split
            return make_regions(filtered_df[filtered_df['lat_proj']<split], min_leaf, region_left)+\
                    make_regions(filtered_df[filtered_df['lat_proj']>=split], min_leaf, region_right)
        else:
            split = filtered_df['lon_proj'].median()
            region_left = region.copy()
            region_right = region.copy()
            region_left[3] = split
            region_right[1] = split
            return make_regions(filtered_df[filtered_df['lon_proj']<split], min_leaf, region_left)+\
                    make_regions(filtered_df[filtered_df['lon_proj']>=split], min_leaf, region_right)
    return [region]


regions = make_regions(filtered_df, min_leaf=100)

#and prepare geopandas dataframe

regions_df = pd.DataFrame(regions, columns=['min_lat', 'min_lon', 'max_lat', 'max_lon'])
regions_df['region_id'] = regions_df.index
regions_df['POLYGON'] = regions_df.apply(lambda x: 'POLYGON(('+\
                                         str(x['min_lon'])+' '+str(x['min_lat'])+', '+\
                                         str(x['max_lon'])+' '+str(x['min_lat'])+', '+\
                                         str(x['max_lon'])+' '+str(x['max_lat'])+', '+\
                                         str(x['min_lon'])+' '+str(x['max_lat'])+', '+\
                                         str(x['min_lon'])+' '+str(x['min_lat'])+\
                                         '))', axis=1)
regions_df['geometry'] = geopandas.GeoSeries.from_wkt(regions_df['POLYGON'])

regions_df = geopandas.GeoDataFrame(regions_df, geometry='geometry')
regions_df.plot(column='region_id', legend=True, cmap='OrRd')

#4 match regions to dataframe

def augment_by_region(filtered_df, regions_df):
    filtered_df_ = geopandas.sjoin(filtered_df, regions_df[['region_id', 'geometry']], how='left')
    #filtered_df.plot(column='price_per_sqm', legend=True, cmap='OrRd')
    #del filtered_df_['geometry']
    del filtered_df_['index_right']
    return filtered_df_

filtered_df = augment_by_region(filtered_df, regions_df)

#4.5 augment dataset by amenities
amenities = pd.read_parquet('data/amenities.parquet')
amenities = project_coordinates(amenities)
amenities = geopandas.GeoDataFrame(amenities,
                             geometry=geopandas.points_from_xy(amenities['lon_proj'],
                                                               amenities['lat_proj']))

top_amenities=['parking', 'restaurant', 'fast_food', 'post_box',
               'kindergarten', 'cafe', 'pub', 'school',
               'place_of_worship', 'doctors', 'pharmacy', 'bank',
               'fuel']
amen_cols = [x+'_avail' for x in top_amenities]

def augment_by_amenities(filtered_df, amenities, top_amenities):
    for amenity in top_amenities:
        t = geopandas.sjoin_nearest(
            filtered_df[['lat_proj', 'lon_proj', 'geometry']],
            amenities[amenities['tagvalue']==amenity][['lat_proj', 'lon_proj', 'geometry']])
        t['avail'] = 1000/(
            abs(t['lat_proj_left']-t['lat_proj_right'])+
            abs(t['lon_proj_left']-t['lon_proj_right']))
        filtered_df[amenity+'_avail'] = t['avail'].to_list()
    return filtered_df

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


def augment_by_dummies(X_train, predictors=None):
    land = pd.get_dummies(X_train['region_id'], prefix='land')
    land = land.multiply(X_train['total_area']/X_train['number_of_units'], axis="index")
    house = pd.get_dummies(X_train['region_id'], prefix='house')
    house = house.multiply(X_train['living_area'], axis="index")
    ptype = pd.get_dummies(X_train['property_type'], prefix='ptype', drop_first=True)
    ptype = ptype.multiply(X_train['living_area'], axis="index")

    X_train_ = pd.concat([X_train, land, house, ptype], axis=1)
    X_train_['freshness'] = (pd.to_datetime(X_train_['construction_date'])-\
                            datetime.fromisocalendar(1800, 1, 1)).apply(
                                lambda x: x.days)
    X_train_['freshness'] = X_train_['freshness'] / 60_000
    X_train_['freshness'] = X_train_['freshness'] * X_train['living_area']

    land_cols = land.columns.to_list()
    house_cols = house.columns.to_list()
    ptypecols = ptype.columns.to_list()
    predictors_ = land_cols + house_cols + ptypecols + amen_cols + ['freshness']
    if predictors is not None:
        for p in set(predictors)-set(predictors_):
            X_train_[p] = 0

    return X_train_, predictors_


X_train, predictors = augment_by_dummies(X_train)

reg = LinearRegression().fit(X_train[predictors], y_train)
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


dt = get_model_score(reg, X_test)

#7 Final test.
'''
Let's use a few generated houses just to ensure
that the whole pipeline is ready to be used
'''
val_data = [
    ['1987-10-19', 'multi_family_house', 5, 100, 2000, 11, 49.45],
    ['1997-10-19', 'single_family_house', 1, 150, 300, 11.1, 49.4],
    ['2007-10-19', 'multi_family_house', 2, 200, 2000, 11.25, 49.46],
    ['2020-10-19', 'single_family_house', 1, 50, 200, 11.2, 49.42],
]
val_df = pd.DataFrame(val_data, columns=['construction_date', 'property_type',
        'number_of_units', 'living_area', 'total_area', 'lon', 'lat',])

val_df = project_coordinates(val_df)
val_df = augment_by_amenities(val_df, amenities, top_amenities)
val_df = augment_by_region(val_df, regions_df)
val_df_dum, predictors_val = augment_by_dummies(val_df, predictors)
val_df['prediction'] = reg.predict(val_df_dum[predictors])
