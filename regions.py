#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 20:03:32 2022

@author: alex
"""
import pandas as pd
import geopandas

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

def make_regions_df(filtered_df):
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
    return regions_df

def visualize_regions(regions_df):
    regions_df['region_id'] = regions_df['region_id']%10*10+regions_df['region_id']//10
    regions_df.plot(column='region_id', legend=True)

def augment_by_region(filtered_df, regions_df):
    filtered_df_ = geopandas.sjoin(filtered_df, regions_df[['region_id', 'geometry']], how='left')
    #filtered_df.plot(column='price_per_sqm', legend=True, cmap='OrRd')
    #del filtered_df_['geometry']
    del filtered_df_['index_right']
    return filtered_df_
