# Data-and-Model-Visualization
Visualizing the prediction of global temperature for a day

### Task
Create a model for predicting the temperature for the next day using the given data. Perform a deeper analysis of the same temperature data utilizing a simplified version of the model you already have.

### Data
The weather data spans a large time period and covers many stations around the globe. There are many possible questions to study. Use a python plotting library of your choice, such as matplotlib.

### Model
The model uses 'latitude', 'longitude', 'elevation', 'yesterday_tmax', 'day_of_year' as input features to predict t_max. Please retrain your model to only use 'latitude', 'longitude', 'elevation', 'day_of_year' before proceeding with task (b) below, and include this re-trained weather-model in your submission.

### Tasks

a) Produce one or more figures that illustrate the daily max. temperature distribution over the entire globe and enable a comparison of different, non-overlapping time periods, e.g. to reveal temporal trends over longer time periods or recurring seasons.

Only show temperatures where you have data available. Take care to handle overplotting of multiple different values into the same point on the figure, which might happen when you have multiple measurements for the same station in a chosen period. By handle overplotting we mean, for instance, to aggregate your data to have a clear meaning for the value that is displayed for a particular station, such as max. or average within the period.
Here is an example from the web:
<img src="https://camo.githubusercontent.com/57b7c235e1d63bab4eb85a7f5c1a349343ea0e3e/687474703a2f2f6333686561646c696e65732e747970657061642e636f6d2f2e612f366130313035333662353830333539373063303133343836653563356536393730632d7069"> </img>

b) b) Produce two or more figures that show the result of your re-trained regression model from CMPT 732-A9, i.e. a version of the model that does not use yesterday_tmax as extra input feature:

(b1) Evaluate your model at a grid of latitude, longitude positions around the globe spanning across oceans and continents, leading to a dense plot of temperatures. This could, for instance, look something like the following:
<img src="https://camo.githubusercontent.com/9f7f661e7e7084f727060a720ad5137af865da30/687474703a2f2f7777772e706879736963616c67656f6772617068792e6e65742f66756e64616d656e74616c732f696d616765732f6a616e5f74656d702e676966">
</img>

You can use a fixed day_of_year of your choice. Also, see further hints about elevation below.

(b2) In a separate plot show the regression error of your model predictions against test data. In this case only use locations where data is given, i.e. you may reuse your plotting method from Part 2 (a).


### Comments and Hints

Any imperfections of your trained model that show up in the visualization are fine. In fact, in this example it is a sign of a good visualization, if it enables us to understand shortcomings of the model. You are not marked for the performance of your model from 732-A9 again, but rather for the methods you create here to investigate it.

Please attempt to make continent or country borders visible on your map. You can do that either by using library function or by using enough data points, such that the shape of some continents roughly emerges from the data distribution. Out of the different datasets please use one with at least 100k rows.

For (b1) you will need elevation information for the points you produce. Have a look at elevation_grid.py for a possible way to add this info to your choice of coordinates. If you place the accompanying elevation data in the same folder as the script you can import the module and see help(evaluation_grid) for example usage.
<img src="https://github.com/sfu-db/bigdata-cmpt733/raw/7cd8078bacf26e3b668f6c73ff9c93b844cee3e9/Assignments/A3/img/elevations.png"> </img>
