#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 11:03:44 2022

@author: alex khablov
"""

import pandas as pd
import numpy as np
import geopandas

from sklearn.model_selection import train_test_split

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


from pyproj import Transformer

transformer = Transformer.from_crs('WGS84', 32632, always_xy=True)


def project_coordinates(data: pd.DataFrame) -> pd.DataFrame:
    """Projects coordinates to meters."""
    lon_proj, lat_proj = transformer.transform(data['lon'].to_numpy(),
                                               data['lat'].to_numpy())
    return data.assign(**{
        'lon_proj': lon_proj,
        'lat_proj': lat_proj,
    })

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

filtered_df = geopandas.GeoDataFrame(filtered_df,
                             geometry=geopandas.points_from_xy(filtered_df['lon_proj'],
                                                               filtered_df['lat_proj']))
filtered_df = geopandas.sjoin(filtered_df, regions_df[['region_id', 'geometry']], how='left')
filtered_df.plot(column='price_per_sqm', legend=True, cmap='OrRd')
del filtered_df['geometry']
del filtered_df['index_right']

#4.5 augment dataset by amenities
amenities = pd.read_parquet('data/amenities.parquet')

amenities = project_coordinates(amenities)

top_amenities=['parking', 'restaurant', 'fast_food', 'post_box',
               'kindergarten', 'cafe', 'pub', 'school',
               'place_of_worship', 'doctors', 'pharmacy', 'bank',
               'fuel']
for amenity in top_amenities:
    t = filtered_df[['lat_proj', 'lon_proj']].join(
        amenities[amenities['tagvalue']==amenity][['lat_proj', 'lon_proj']],
        how='cross',
        rsuffix='_a')
    t['avail'] = 1000/(abs(t['lat_proj']-t['lat_proj_a'])+abs(t['lon_proj']-t['lon_proj_a']))
    t = t.groupby(['lat_proj', 'lon_proj'], as_index=False)['avail'].max()
    t.columns=['lat_proj', 'lon_proj', amenity+'_avail']
    filtered_df = pd.merge(filtered_df,t, on=['lat_proj', 'lon_proj'])

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

amen_cols = [x+'_avail' for x in top_amenities]
X_train, X_test, y_train, y_test = train_test_split(
    filtered_df[['region_id', 'living_area', 'total_area', 'number_of_units',
                 'property_type']+amen_cols],
    filtered_df['price'], test_size=0.33, random_state=42)

land = pd.get_dummies(X_train['region_id'], prefix='land')
land = land.multiply(X_train['total_area']/X_train['number_of_units'], axis="index")
house = pd.get_dummies(X_train['region_id'], prefix='house')
house = house.multiply(X_train['living_area'], axis="index")
ptype = pd.get_dummies(X_train['property_type'], prefix='ptype', drop_first=True)
ptype = ptype.multiply(X_train['living_area'], axis="index")

land_cols = land.columns.to_list()
house_cols = house.columns.to_list()
ptypecols = ptype.columns.to_list()
predictors = land_cols + house_cols + ptypecols + amen_cols

X_train = pd.concat([X_train, land, house, ptype], axis=1)

from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train[predictors], y_train)
print(reg.score(X_train[predictors], y_train))
feaure_coeffs = dict(zip(predictors, reg.coef_))
print(feaure_coeffs)



land = pd.get_dummies(X_test['region_id'], prefix='land')
land = land.multiply(X_test['total_area']/X_test['number_of_units'], axis="index")
house = pd.get_dummies(X_test['region_id'], prefix='house')
house = house.multiply(X_test['living_area'], axis="index")
ptype = pd.get_dummies(X_test['property_type'], prefix='ptype', drop_first=True)
ptype = ptype.multiply(X_test['living_area'], axis="index")

X_test = pd.concat([X_test, land, house, ptype], axis=1)

print(reg.score(X_test[predictors], y_test))

X_test['prediction'] = reg.predict(X_test[predictors])
X_test['price'] = y_test
X_test['err'] = X_test['prediction'] - X_test['price']
X_test['abs_err'] = abs(X_test['err'])
X_test['err_prcnt'] = X_test['err']/X_test['price']
X_test['abs_err_prcnt'] = abs(X_test['err_prcnt'])

dt = X_test[['prediction', 'price', 'err', 'err_prcnt', 'abs_err', 'abs_err_prcnt']].describe().T
