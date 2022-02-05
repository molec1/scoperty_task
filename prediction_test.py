#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 20:30:33 2022

@author: alex
"""
import pandas as pd
import geopandas
import pickle
from datetime import datetime

from projections import project_coordinates
from amenities import augment_by_amenities, top_amenities, get_amentities
from regions import augment_by_region
from augmentation import augment_by_dummies

def prediction_test():
    print(datetime.now())
    amenities = get_amentities()
    reg = pickle.load(open('model/model.sav', 'rb'))
    predictors = pickle.load(open('model/predictors.sav', 'rb'))
    regions_df = geopandas.read_parquet('model/regions.parquet')
    print(datetime.now())
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
    print(datetime.now())
    return val_df
