from types import LambdaType
import yfinance as yf 
import pandas as pd 
import numpy as np
from ta import add_all_ta_features
import fastai.tabular
import plotly.graph_objs as go  # Import the graph ojbects
from sklearn.metrics import mean_squared_error # Install error metrics 
from sklearn.linear_model import LinearRegression # Install linear regression model
from sklearn.neural_network import MLPRegressor # Install ANN model 
from sklearn.preprocessing import StandardScaler # to scale for ann
import matplotlib.pyplot as plt


tickerSymbol = "TSLA"
startDate = '2019-11-18'
# endDate = '2021-11-18'

# tickerData = yf.Ticker(tickerSymbol)

# df = tickerData.history(start = startDate)#, end = endDate)

df = yf.download(tickerSymbol, period='5d', interval='1m')

# date_change = '%Y-%m-%d'
# df['Date'] = df.index
# df['Date'] = pd.to_datetime(df['Date'], format = date_change)
# Dates = df['Date']
# df = add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume", fillna=True) 
# fastai.tabular.add_datepart(df, 'Date', drop = 'True')
# df['Date'] = pd.to_datetime(df.index.values, format = date_change)
# fastai.tabular.add_cyclic_datepart(df, 'Date', drop = 'True')

shifts = [1, 5, 10]
train_pct = .75

w = 16
h = 4 

def CorrectColumnTypes(df):
  for col in df.columns[1:80]: df[col] = df[col].astype('float')
  for col in df.columns[-10:]: df[col] = df[col].astype('float')
  for col in df.columns[80:-10]: df[col] = df[col].astype('category')
  return df

def CreateLags(df, lag_size):
  shiftdays = lag_size
  shift = -shiftdays
  df['Close_lag'] = df['Adj Close'].shift(shift)
  return df, shift

def SplitData(df, train_pct, shift):
  train_pt = int(len(df)*train_pct)
  train = df.iloc[:train_pt,:]
  test = df.iloc[train_pt:,:]
  x_train = train.iloc[:shift,1:-1]
  y_train = train['Close_lag'][:shift]
  x_test = test.iloc[:shift,1:-1]
  y_test = test['Adj Close'][:shift]
  return x_train, y_train, x_test, y_test, train, test

def plot(train, test, pred, ticker, w, h, shift_days, name):
  # plt.plot(train.index, train['Adj Close'], label = 'Train Actual')
  # try: plt.plot(test.index[:shift], test['Adj Close'][:-1], label = 'Test Actual')
  # except:
  #   length_diff = len(test.index[:shift])-len(test['Close'])
  #   plt.plot(test.index[:shift], test['Adj Close'][:length_diff], label = 'Test Actual')

  # plt.plot(test.index[:shift], pred, label = 'Our Prediction')

  plt.plot(list(range(len(train['Adj Close']), len(train['Adj Close']))), train['Adj Close'], label = 'Train Actual')
  try: plt.plot(list(range(len(train['Adj Close']), len(train['Adj Close'])+len(test.index[:shift]))), test['Adj Close'], label = 'Test Actual')
  except:
    length_diff = len(test.index[:shift])-len(test['Adj Close'])
    plt.plot(list(range(len(train['Adj Close']), len(train['Adj Close'])+len(test.index[:shift]))), test['Adj Close'][:length_diff], label = 'Test Actual')
  plt.plot(list(range(len(train['Adj Close']), len(train['Adj Close'])+len(test.index[:shift]))), pred, label = 'Our Prediction')
  
  plt.xticks(rotation=75)
  plt.legend()
  plt.show()

def PlotModelResults_Plotly(train, test, pred, ticker, w, h, shift_days, name):
  D1 = go.Scatter(x=train.index, y=train['Close'], name = 'Train Actual')
  D2 = go.Scatter(x=test.index[:shift], y=test['Close'], name = 'Test Actual')
  D3 = go.Scatter(x=test.index[:shift], y=pred, name = 'Our Prediction')
  line = {'data': [D1,D2,D3],
          'layout': {
              'xaxis' :{'title': 'Date'},
              'yaxis' :{'title': '$'},
              'title' : name + ' - ' + ticker + ' - ' + str(shift_days)
          }}
  fig = go.Figure(line)
  fig.show()

def LinearRegression_fnc(x_train,y_train, x_test, y_test):
  lr = LinearRegression()
  lr.fit(x_train,y_train)
  lr_pred = lr.predict(x_test)
  lr_MSE = mean_squared_error(y_test, lr_pred)
  lr_R2 = lr.score(x_test, y_test)
  print('Linear Regression R2: {}'.format(lr_R2))
  print('Linear Regression MSE: {}'.format(lr_MSE))
  return lr_pred

def ANN_func(x_train,y_train, x_test, y_test):
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

def CalcProfit(test_df, pred, j):
  pd.set_option('mode.chained_assignment', None)
  test_df['pred'] = np.nan
  test_df['pred'].iloc[:-j] = pred
  test_df['change'] = test_df['Close_lag'] - test_df['Adj Close'] 
  test_df['change_pred'] = test_df['pred'] - test_df['Adj Close'] 
  test_df['MadeMoney'] = np.where(test_df['change_pred']/test_df['change'] > 0, 1, -1) 
  test_df['profit'] = np.abs(test['change']) * test_df['MadeMoney']
  profit_dollars = test['profit'].sum()
  print('Would have made: $ ' + str(round(profit_dollars,1)))
  profit_days = len(test_df[test_df['MadeMoney'] == 1])
  print('Percentage of good trading days: ' + str(round(profit_days/(len(test_df)-j),2))     )
  return test_df, profit_dollars

for j in shifts: 
  print(str(j) + ' days out:')
  print('------------')
  df_lag, shift = CreateLags(df,j)
  df_lag = CorrectColumnTypes(df_lag)
  x_train, y_train, x_test, y_test, train, test = SplitData(df, train_pct, shift)
  test += abs(train['Adj Close'][-1]-test['Adj Close'][0])

  # Linear Regression
  print("Linear Regression")
  lr_pred = LinearRegression_fnc(x_train, y_train, x_test, y_test)
  lr_pred += abs(train['Adj Close'][-1]-lr_pred[0])
  test2, profit_dollars = CalcProfit(test, lr_pred,j)
  plot(train, test, lr_pred, tickerSymbol, w, h, j, 'Linear Regression')
  # exit(1)
  # Artificial Neuarl Network 
  print("ANN")
  MLP_pred = ANN_func(x_train, y_train, x_test, y_test)
  MLP_pred += abs(train['Adj Close'][-1]-MLP_pred[0])
  test2, profit_dollars = CalcProfit(test, MLP_pred, j)
  plot(train, test, MLP_pred, tickerSymbol, w, h, j, 'ANN')
  print('------------')
