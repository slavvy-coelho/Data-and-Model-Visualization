#running instructions: login to the cluster
#                      unzip and upload the model (weather_model)to hdfs, 
#                           command: 1. unzip weather_model.zip
#                                    2. hadoop fs -put weather_model
#running command: time spark-submit weather_plot.py /courses/732/tmax-2 /courses/732/tmax-test weather_model
import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('weather prediction').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
assert spark.version >= '2.3' # make sure we have Spark 2.3+

from pyspark.ml.linalg import Vectors #for model 
from pyspark.sql.functions import dayofyear
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator

from mpl_toolkits.basemap import Basemap, shiftgrid #for plotting maps
from matplotlib import colors as c
from itertools import chain

import elevation_grid as eg #importing the provided code

#************************************************************************

def main(tmax2, tmax_test, model1):
	#******************* Task A **********************************
	#defining schema
    tmax_schema = types.StructType([
    types.StructField('station', types.StringType()),
    types.StructField('date', types.DateType()),
    types.StructField('latitude', types.FloatType()),
    types.StructField('longitude', types.FloatType()),
    types.StructField('elevation', types.FloatType()),
    types.StructField('tmax', types.FloatType()),
])
    temp_data = spark.read.csv(tmax2, schema=tmax_schema)
    temp_data.registerTempTable("table") #temporary view

    group_1 = spark.sql("""SELECT station, latitude, longitude, avg(tmax) AS old_temp 
    						FROM table
    						WHERE year(date) < 2000
    						GROUP BY station, latitude, longitude""")
    group_1.registerTempTable("group_1")

    group_2 = spark.sql("""SELECT station, latitude, longitude, avg(tmax) AS new_temp 
    						FROM table
    						WHERE year(date) >= 2000
    						GROUP BY station, latitude, longitude""")
    group_2.registerTempTable("group_2")

    avg_temp_diff = spark.sql("""SELECT A.station, A.latitude, A.longitude, (B.new_temp-A.old_temp) AS Avg_temp 
                                    FROM group_1 A, group_2 B 
                                    WHERE A.station == B.station 
                                    ORDER BY Avg_temp ASC""")

    #list of points to be plotted
    lat_list = avg_temp_diff.select("latitude").rdd.flatMap(lambda x: x).collect()
    lon_list = avg_temp_diff.select("longitude").rdd.flatMap(lambda x: x).collect()
    diff_list = avg_temp_diff.select("Avg_temp").rdd.flatMap(lambda x: x).collect()

    #projecting on the world map
    #1. creating the map
	    # llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon are the lat/lon values of the lower left and upper right corners
		# of the map.
		# lat_ts is the latitude of true scale.
		# resolution = 'c' means use crude resolution coastlines.

    m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
            llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c') 
    m.bluemarble(scale=0.2)
    m.drawcoastlines(color='white', linewidth=0.2)
    m.drawparallels(np.arange(-90.,91.,30.), labels=[1,0,0,0]) # draw parallels and meridians.
    m.drawmeridians(np.arange(-180.,181.,60.), labels=[0,0,0,1])
	
	#2. plotting data
    cmap = c.ListedColormap(['#008744','#0057e7','#d62d20','#ffa700',
                            '#ff0000','#004cff',
                             '#0073ff','#0099ff','#00c0ff','#00d900','#33f3ff','#73ffff','#c0ffff', 
                                '#ff9900','#ff8000','#ff6600',
                             '#ff4c00','#ff2600','#e60000','#b30000','#800000','#4c0000'])

    mxy = m(lon_list, lat_list)
    m.scatter(mxy[0], mxy[1], s=12, c= diff_list,cmap=cmap, lw=0, alpha=1, zorder=5)
    plt.figure(1, figsize=(30, 20))
    plt.title("Temperature difference across years")
    plt.colorbar(orientation = 'horizontal')
    plt.show()
    plt.savefig('plot1_new.png')
    plt.clf()

    #********************************** Task B1 *****************************************
    lats, lons = np.meshgrid(np.arange(-90,90,.5),np.arange(-180,180,.5))
    elevs = [eg.get_elevations(np.array([late,lone]).T) for late,lone in zip(lats,lons)]


    lat_df = pd.DataFrame(lats.flatten())
    lon_df = pd.DataFrame(lons.flatten())
    merged = list(itertools.chain(*elevs))
    new_merged = pd.DataFrame(merged)


    frames = [lat_df, lon_df, new_merged]
    result = pd.concat(frames, axis=1)
    result.columns = ['latitude','longitude','elevation']
    result['date'] = '2018-02-05'
    result['tmax'] = 'O'

    convert_sparkdf = spark.createDataFrame(result)
    tmax_model = model1

    # load the model
    model = PipelineModel.load(tmax_model)

    # use the model to make predictions
    predictions = model.transform(convert_sparkdf)
    tmax_pred = predictions.select("prediction").rdd.flatMap(lambda x: x).collect()

    m1 = Basemap(projection='robin', lon_0=0, resolution='c')
    cmap = c.ListedColormap(['#67001f','#b2182b', '#d6604d','#f4a582','#fddbc7','#f7f7f7','#d1e5f0','#92c5de','#4393c3','#2166ac','#053061'])

    m1.drawparallels(np.arange(-90.,120.,30.),labels=[1,0,0,0])
    m1.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])
    m1.drawcountries(linewidth=0.1, color="w")              
    m1.drawcoastlines(color='white', linewidth=0.2)
    
    # Plot the data
    mxy = m1(lons.flatten(), lats.flatten())
    m1.scatter(mxy[0], mxy[1], c=tmax_pred,cmap=cmap, s=8, lw=0, alpha=1)
    
    plt.figure(1, figsize=(35,20))
    plt.colorbar(orientation = 'horizontal')
    plt.title("Density plot of temperature over the years")
    plt.show()
    plt.savefig('plot2_new.png')
    plt.clf()

    #********************************** Task B2 *****************************************
    # get the data
    tmax_2 = spark.read.csv(tmax_test, schema=tmax_schema)

    # load the model
    tmax_model = model1
    model = PipelineModel.load(tmax_model)

    # use the model to make predictions
    new_predictions = model.transform(tmax_2)

    diff = new_predictions.withColumn('difference', functions.abs(new_predictions['tmax'] - new_predictions['prediction']))
    new_diff = diff.select("difference").rdd.flatMap(lambda x: x).collect()
    lat_new = diff.select("latitude").rdd.flatMap(lambda x: x).collect()
    lon_new = diff.select("longitude").rdd.flatMap(lambda x: x).collect()

    m2 = Basemap(projection='robin', lon_0=0, resolution='c')
    cmap = c.ListedColormap(['#67001f','#b2182b', '#d6604d','#f4a582','#fddbc7','#f7f7f7','#d1e5f0','#92c5de','#4393c3','#2166ac','#053061'])

    m2.drawparallels(np.arange(-90.,120.,30.),labels=[1,0,0,0])
    m2.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])
    m2.drawcountries(linewidth=0.1, color="w")              
    m2.drawcoastlines(color='white', linewidth=0.2)
    
    # Plot the data
    mxy = m2(lon_new, lat_new)
    m2.scatter(mxy[0], mxy[1], c = new_diff, cmap=cmap, s=8, lw=0, alpha=1)
    
    plt.figure(1, figsize=(30,18))
    plt.colorbar(orientation = 'horizontal')
    plt.title("Temperature difference between the trained model and the original values")
    plt.show()
    plt.savefig('plot3_new.png')



if __name__ == '__main__':
    tmax2 = sys.argv[1]
    tmax_test = sys.argv[2]
    model = sys.argv[3]
    main(tmax2, tmax_test, model)

