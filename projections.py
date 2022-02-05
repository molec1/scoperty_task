#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 20:13:18 2022

@author: alex
"""
import pandas as pd
import geopandas
from pyproj import Transformer


def project_coordinates(data: pd.DataFrame) -> pd.DataFrame:
    """Projects coordinates to meters."""
    transformer = Transformer.from_crs('WGS84', 32632, always_xy=True)
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
