# -*- coding: utf-8 -*-
"""Predict The Stock Market With Python Just Code.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/gist/AveryData/d6abae0b23bc310ef39de34d37e452ba/predict-the-stock-market-with-python-just-code.ipynb

# Now We Will Start On The Code

We'll start by getting code to install Python packages on Google Collab.

We'll be using the following libraries:

*    pandas 
*    numpy 
*    ta
*    yfinance
*    plotly
"""

# Code to do library installations
import sys # Import the library that does system activities (like install other packages)

"""Next we'll have code that you, as a user, will input. 

You'll submit what company ticker you're interested in, and the start and end dates of interest. 

You can tind the ticker symbols for a lot of companies [here](http://www.eoddata.com/symbols.aspx?AspxAutoDetectCookieSupport=1).

When selected a date range, keep in mind COVID-19 has had an insane effect on the market. Stocks are trading very irreguarly and differently than the did, pre-COVID. If you traing on historical data before COVID, and try to use it after COVID, your model might not be that effective. 
"""

# Choose your ticker
tickerSymbol = "AMZN"

# Choose date range - format should be 'YYYY-MM-DD' 
startDate = '2015-04-01' # as strings
endDate = '2020-01-01' # as strings

"""Next, we'll go ahead and install that *yfinance* Python library. As a reminder, this is how we'll get stock price information from the Yahoo! Finance website. 

This will be what we use to go and get the stock data for that ticker.
"""

# Check if local computer has the library yfinance. If not, install. Then Import it.
# !{sys.executable} -m pip install yfinance # Check if the machine has yfinance, if not, download yfinance
import yfinance as yf # Import library to access Yahoo finance stock data

"""Now that *yfinance* is imported, let's go ahead and get the stock data using the *yfinance* package.

We'll print out a preview of what the data looks like once it is complete. 
"""

# Create ticker yfinance object
tickerData = yf.Ticker(tickerSymbol)

# Create historic data dataframe and fetch the data for the dates given. 
df = tickerData.history(start = startDate, end = endDate)

# Print statement showing the download is done

# Show what the first 5 rows of the data frame
# Note the dataframe has:
#   - Date (YYY-MM-DD) as an index
#   - Open (price the stock started as)
#   - High (highest price stock reached that day)
#   - Low (lowest price stock reached that day)
#   - Close (price the stock ended the day as)
#   - Volume (how many shares were traded that day)
#   - Dividends (any earnings shared to shareholders)
#   - Stock Splits (any stock price changes)

print('-----------------------')
print('Done!')
print(df.head())

"""Let's get another useful library imported, [pandas](https://pandas.pydata.org/). 

*pandas* is the best way to manipulate dataframe objects.

[*numpy*](https://numpy.org/) is also helpful dealing with data structures. 
"""

# Import the library that does dataframe management
import pandas as pd # Library that manages dataframes
import numpy as np

"""The date is just a string right now, but Python is smart and can realize it is a date if we help it out. These date variable types are easier to work with and efficient. 

Let's change the date time from a string to a date type. 
"""

# Change the date column to a pandas date time column 

# Define string format
date_change = '%Y-%m-%d'

# Create a new date column from the index
df['Date'] = df.index

# Perform the date type change
df['Date'] = pd.to_datetime(df['Date'], format = date_change)

# Create a variable that is the date column
Dates = df['Date']

"""We know the "Open", "High", "Low", "Close", "Volume" are useful, but there is more data that can be derived off of this data. 

Financial Technical Indicators are useful to understand what is going on with a particular stock. 

We will create some of these with help from a package called *ta* standing for technical anlysis. 

First, we'll have to install and then import the package.
"""

# Add financial information and indicators 
# !{sys.executable} -m pip install ta # Download ta
from ta import add_all_ta_features # Library that does financial technical analysis

"""Now that the package is imported, let's add these technical indicators to our dataframe.

We'll print out each column name of our dataframe to see what new columns we gained. 
"""

# Add all technical analysis to the dataframe we've already loaded
df = add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume", fillna=True) 

print(df.columns)

"""Yay! Now, we've added the techincal indicators!

You can learn and understand what all these new values are on the documentation of the *ta* site. They have a [dictionary](https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html) that exmplains what these indicators are, and what they mean. 

Now that we have the technical indicators and dates sorted out, let's add some date features that will show what month it is, what day of the dear it is, what day in the quarter it is, ect. 

To do that, we will use a Python package called *fastai*. 

Let's install the package.
"""

# Install fastai to use the date function
# !{sys.executable} -m pip install fastai # Download fastai 
import fastai.tabular # Library that does date factors

"""After it is imported, let's add the new date features. """

# Define the date parts 
fastai.tabular.add_datepart(df,'Date', drop = 'True')

# Ensure the correct format
df['Date'] = pd.to_datetime(df.index.values, format = date_change)

# Add the date parts
fastai.tabular.add_cyclic_datepart(df, 'Date', drop = 'True')

"""### Data Pulling, Complete!

We've now pulled all the data we need. We'll start creating our model now! 

Let's start by defining how far out we want to do our predictions. 

I'm interested in 1 day out, 5 days out, and 10 days out so I'll add those to my *shifts* list. 

We'll also define how much of our data we want to use to train and how much we will use to evaluate the model. 75% is a good start. 



"""

# Define key model parameters

# Set days out to predict 
shifts = [1,5,10]

# Set a training percentage
train_pct = .75

# Plotting dimensions
w = 16 # width
h = 4 # height

"""### Defining Functions

Next, we'll define some functions to do some tasks for us.

The first one is boring and tedious, but the packages we used were a little lazy on what variable types they used. The following function just goes through and makes sure the right columns are numbers (floats) and the right columns are categories (like strings). 
"""

# Ensure column types are correct

def CorrectColumnTypes(df):
  # Input: dataframe 
  # ouptut: dataframe (with column types changed)

  # Numbers
  for col in df.columns[1:80]:
      df[col] = df[col].astype('float')

  for col in df.columns[-10:]:
      df[col] = df[col].astype('float')

  # Categories 
  for col in df.columns[80:-10]:
      df[col] = df[col].astype('category')

  return df

"""In order to do the days in the future, we have to move our closing costs by that number of days.

We'll write a function that does that for us. 
"""

# Create the lags 
def CreateLags(df,lag_size):
  # inputs: dataframe , size of the lag (int)
  # ouptut: dataframe ( with extra lag column), shift size (int)

  # add lag
  shiftdays = lag_size
  shift = -shiftdays
  df['Close_lag'] = df['Close'].shift(shift)
  return df, shift

"""Finally, we'll actually divide the historic data into the test and train sets.

We'll split up the x's and the y as well for this.

We'll end up with a test and training set for the *x*'s and the *y*. 
"""

# Split the testing and training data 
def SplitData(df, train_pct, shift):
  # inputs: dataframe , training_pct (float between 0 and 1), size of the lag (int)
  # ouptut: x train dataframe, y train data frame, x test dataframe, y test dataframe, train data frame, test dataframe

  train_pt = int(len(df)*train_pct)
  
  train = df.iloc[:train_pt,:]
  test = df.iloc[train_pt:,:]
  
  x_train = train.iloc[:shift,1:-1]
  y_train = train['Close_lag'][:shift]
  x_test = test.iloc[:shift,1:-1]
  y_test = test['Close'][:shift]

  return x_train, y_train, x_test, y_test, train, test

"""The best way to understand how good our predictions are is to actually *see* and *compare*. We'll do this by making a time series visualization.This visual will compare the actual versus the predicted over time. 

The best visualization package for Python is [plotly](https://plotly.com/). 

We'll start by installing it. 
"""

# !{sys.executable} -m pip install plotly # Download plotly 
import plotly.graph_objs as go  # Import the graph ojbects

"""Now we'll make a function that greats a sweet graph for us"""

# Function to make the plots
def PlotModelResults_Plotly(train, test, pred, ticker, w, h, shift_days,name):
  # inputs: train dataframe, test dataframe, predicted value (list), ticker ('string'), width (int), height (int), shift size (int), name (string)
  # output: None

  # Create lines of the training actual, testing actual, prediction 
  D1 = go.Scatter(x=train.index,y=train['Close'],name = 'Train Actual') # Training actuals
  D2 = go.Scatter(x=test.index[:shift],y=test['Close'],name = 'Test Actual') # Testing actuals
  D3 = go.Scatter(x=test.index[:shift],y=pred,name = 'Our Prediction') # Testing predction

  # Combine in an object  
  line = {'data': [D1,D2,D3],
          'layout': {
              'xaxis' :{'title': 'Date'},
              'yaxis' :{'title': '$'},
              'title' : name + ' - ' + tickerSymbol + ' - ' + str(shift_days)
          }}
  # Send object to a figure 
  fig = go.Figure(line)

  # Show figure
  fig.show()

"""## Making the Model

In order to make the models, we'll be using a package called SciKit Learn.

We'll have to install and import the package. 
"""

# Import sklearn modules that will help with modeling building

# !{sys.executable} -m pip install sklearn # Download sklearn 
from sklearn.metrics import mean_squared_error # Install error metrics 
from sklearn.linear_model import LinearRegression # Install linear regression model
from sklearn.neural_network import MLPRegressor # Install ANN model 
from sklearn.preprocessing import StandardScaler # to scale for ann

"""As discussed earlier, the easiest form of machine learning is linear regression. """

# Regreesion Function

def LinearRegression_fnc(x_train,y_train, x_test, y_test):
  #inputs: x train data, y train data, x test data, y test data (all dataframe's)
  # output: the predicted values for the test data (list)
  
  lr = LinearRegression()
  lr.fit(x_train,y_train)
  lr_pred = lr.predict(x_test)
  lr_MSE = mean_squared_error(y_test, lr_pred)
  lr_R2 = lr.score(x_test, y_test)
  print('Linear Regression R2: {}'.format(lr_R2))
  print('Linear Regression MSE: {}'.format(lr_MSE))

  return lr_pred

# ANN Function 

def ANN_func(x_train,y_train, x_test, y_test):

  # Scaling data
  scaler = StandardScaler()
  scaler.fit(x_train)
  x_train_scaled = scaler.transform(x_train)
  x_test_scaled = scaler.transform(x_test)


  MLP = MLPRegressor(random_state=1, max_iter=1000, hidden_layer_sizes = (100,), activation = 'identity',learning_rate = 'adaptive').fit(x_train_scaled, y_train)
  MLP_pred = MLP.predict(x_test_scaled)
  MLP_MSE = mean_squared_error(y_test, MLP_pred)
  MLP_R2 = MLP.score(x_test_scaled, y_test)

  print('Muli-layer Perceptron R2 Test: {}'.format(MLP_R2))
  print('Multi-layer Perceptron MSE: {}'.format(MLP_MSE))

  return MLP_pred

"""Let's create one last function to calculate how much money we would have made, had we been trading this strategy"""

def CalcProfit(test_df,pred,j):
  pd.set_option('mode.chained_assignment', None)
  test_df['pred'] = np.nan
  test_df['pred'].iloc[:-j] = pred
  test_df['change'] = test_df['Close_lag'] - test_df['Close'] 
  test_df['change_pred'] = test_df['pred'] - test_df['Close'] 
  test_df['MadeMoney'] = np.where(test_df['change_pred']/test_df['change'] > 0, 1, -1) 
  test_df['profit'] = np.abs(test['change']) * test_df['MadeMoney']
  profit_dollars = test['profit'].sum()
  print('Would have made: $ ' + str(round(profit_dollars,1)))
  profit_days = len(test_df[test_df['MadeMoney'] == 1])
  print('Percentage of good trading days: ' + str( round(profit_days/(len(test_df)-j),2))     )

  return test_df, profit_dollars

"""## Let's Start Predicting!
## Time To Make Money!

We've gotten our data, created functions, now let's get to the point of actually doing predictions. 

For the ticker, we'll have a prediction for each time length out into the future. 
"""

# Go through each shift....

for j in shifts: 
  print(str(j) + ' days out:')
  print('------------')
  df_lag, shift = CreateLags(df,j)
  df_lag = CorrectColumnTypes(df_lag)
  x_train, y_train, x_test, y_test, train, test = SplitData(df, train_pct, shift)

  # Linear Regression
  print("Linear Regression")
  lr_pred = LinearRegression_fnc(x_train,y_train, x_test, y_test)
  test2, profit_dollars = CalcProfit(test,lr_pred,j)
  PlotModelResults_Plotly(train, test, lr_pred, tickerSymbol, w, h, j, 'Linear Regression')

  # Artificial Neuarl Network 
  print("ANN")
  MLP_pred = ANN_func(x_train,y_train, x_test, y_test)
  test2, profit_dollars = CalcProfit(test,MLP_pred,j)
  PlotModelResults_Plotly(train, test, MLP_pred, tickerSymbol, w, h, j, 'ANN')
  print('------------')