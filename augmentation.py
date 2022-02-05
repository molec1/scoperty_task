#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 20:18:49 2022

@author: alex
"""
import pandas as pd
from datetime import datetime

from amenities import amen_cols


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

    return X_train_.copy(), predictors_
