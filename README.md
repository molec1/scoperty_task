# scoperty_task
Real estate data sample AVM development task.

## 1. The Data.

This dataset contains only 10k rows, all describing Nurnberg city real estate market.
The dataset contains just 12 rows with apartments transactions data, because of it I decided to exclude it from the training set. We cannot make the apartments model using this amount of data but this data could cause the houses valuation model quality.
Almost all additional columns contained to much empty values.
The trusted data was delivered only for GEO columns + property_type, construction date and price.
I decided to use only this small amount of data without filling the gaps in additional columns.

After exploratory analysis I found a few issues this data. Bad examples were excluded in filter function.

## 2. Coordinates

There are a few approaches how to deal with GEO data as predictor. The DS could use clusterization, predefined administrative regions, hot points on map or other approaches. I decided to cluster the whole map by 64 regions with almost equal number of points inside. These regions are small enough to describe city regions, but big enough to overtrain.
To make these regions as square as it could be I converted the coordinates to meters and used self-written algorithm.

## 3. GEO features augmentation

Just to examplify the approach I've got Nurnberg OSM data, parsed amenities and city buildings from it and store in separated parquet file. This dataset I've joined to the main dataset and for each offer found the nearest school, kindergarten, etc. Using this information I've added a few new features to the dataset. At this time these features are useless, but it was made mainly to illustrate the idea.

## 4. The Construction date

Old buildings very often are cheaper. To use this as predictor I've added the new column freshness, with values from 0.5 to 2, where 0.5 stands for 19th century buildings, 2 stands for the new ones. Because of using very simple model I had to multiply this value by house living_area.

## 5. The modelavm_train.py

I've decided to use ElasticNet this very low alpha and param positive=True to prepare very rich but not overfitted model. Amenities features shows high level of milticolinearity, plus the land in some regions tend to be very cheap.

## 6. Accuracy

Test MAPE is 13.7%, MdAPE = 11.3%, trainf MAPE is 8%. I do believe this level is very good for this simple and quick model.

## 7. Further usage

* The model could be used to provide price maps for web sites, real estate agents or thers.
* Using the price history one could deliver the price forecast for each real estate object.
* Using offer state info it could be calculated, is it price effective to make a restoration of object or not before the sell.
* The model could predict the price of never constructed building before the planning works. 
* It could be calculated, is it cheaper to pay the mortgage instead of rent payments or not.

The main script of the project is **avm_train.py**, all other scripts are imported.
