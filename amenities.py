#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 20:12:18 2022

@author: alex
"""
import pandas as pd
import geopandas

from projections import project_coordinates

def get_amentities():
    amenities = pd.read_parquet('data/amenities.parquet')
    amenities = project_coordinates(amenities)
    amenities = geopandas.GeoDataFrame(amenities,
                                 geometry=geopandas.points_from_xy(amenities['lon_proj'],
                                                                   amenities['lat_proj']))
    return amenities

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
        t['avail'] = t['avail'].apply(lambda x: 5 if x>5 else x)
        filtered_df[amenity+'_avail'] = t['avail'].to_list()
    return filtered_df
