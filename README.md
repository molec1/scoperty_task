# scoperty_task
Real estate data sample AVM development task.

## 1. The Data

This dataset contains only 10k rows, all describing Nurnberg city real estate market.
The dataset contains just 12 rows with apartments transactions data, because of it I decided to exclude it from the training set. We cannot make the apartments’ model using this amount of data, but this data could cause the houses’ valuation model quality to fall.
Almost all additional columns contained too many empty values.
The trusted data was delivered only for GEO columns + property_type, construction date, and price.
I decided to use only this small amount of data, without filling the gaps in additional columns.

After exploratory analysis, I found a few issues in this data. Bad examples were excluded in the filter function.

## 2. Coordinates

There are a few approaches how to deal with GEO data as predictors. The DS could use clusterization, predefined administrative regions, hot points on a map, or other approaches. I decided to cluster the whole map by 64 regions with an almost equal number of points inside. These regions are small enough to describe city regions, but big enough to overtrain.
To make these regions as square as they could be, I converted the coordinates to meters and used a self-written algorithm.

## 3. GEO features augmentation

Just to exemplify the approach, I've got Nurnberg OSM data, parsed amenities and city buildings from it, and stored it in a separated parquet file. This dataset I've joined to the main dataset and for each offer found the nearest school, kindergarten, etc. Using this information, I've added a few new features to the dataset. At this time these features are useless, but it was made mainly to illustrate the idea.

## 4. The Construction date

Old buildings very are often cheaper. To use this feature as a predictor I've added the new column freshness, with values from 0.5 to 2, where 0.5 stands for 19th-century buildings, 2 stands for the new ones. Because of using a very simple model, I had to multiply this value by house living_area.

## 5. The model
I've decided to use ElasticNet with very low alpha and parameter positive=True to prepare a rich but not overfitted model. Amenity features show a high level of multicollinearity, plus the land in some regions tends to be very cheap.

## 6. Accuracy

Test MAPE is 13.7%, MdAPE = 11.3%, train MAPE is 8%. I do believe this level is very good for this simple and quick model.

## 7. Further usage

* The model could be used to provide price maps for websites, real estate agents others.
* Using the price history, one could deliver the price forecast for each real estate object.
* Using offer state info, it could be calculated if the price is effective to make a restoration of an object or not before the sale.
* The model could predict the price of a never-constructed building before the planning works. 
* It could be calculated if it is cheaper to pay the mortgage instead of rent payments or not.

The main script of the project is **avm_train.py**, all other scripts are imported.

