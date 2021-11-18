import yfinance as yf 
import pandas as pd 
import numpy as np
from ta import add_all_ta_features
from sklearn.metrics import mean_squared_error # Install error metrics 
from sklearn.linear_model import LinearRegression # Install linear regression model
from sklearn.neural_network import MLPRegressor # Install ANN model 
from sklearn.preprocessing import StandardScaler # to scale for ann
import matplotlib.pyplot as plt

ticker = "AAPL"
df = yf.download(ticker, period='5y', interval='1d')
training_percentage = .9

def setup_lags(df, lag_size):
  shift = -lag_size
  df['Close_lag'] = df['Adj Close'].shift(shift)
  return df, shift

def seperate_data(df, train_pct, shift):
  train_pt = int(len(df)*train_pct)
  train = df.iloc[:train_pt,:]
  test = df.iloc[train_pt:,:]
  x_train = train.iloc[:shift,1:-1]
  y_train = train['Close_lag'][:shift]
  x_test = test.iloc[:shift,1:-1]
  y_test = test['Adj Close'][:shift]
  return x_train, y_train, x_test, y_test, train, test

def plot(train, test, pred, ticker, shift_days, name):
  plt.plot(train.index, train['Adj Close'], label = 'Train Actual')
  try: plt.plot(test.index[:shift], test['Adj Close'][:-1], label = 'Test Actual')
  except:
    length_diff = len(test.index[:shift])-len(test['Close'])
    plt.plot(test.index[:shift], test['Adj Close'][:length_diff], label = 'Test Actual')
  plt.plot(test.index[:shift], pred, label = 'Our Prediction')
  plt.title(ticker+' - '+name+' - '+str(shift_days))
  plt.xticks(rotation=45)
  plt.legend()
  plt.show()

def LinearRegression_fnc(x_train, y_train, x_test, y_test):
  lr = LinearRegression()
  lr.fit(x_train, y_train)
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
  print('Percentage of good trading days: ' + str(round(profit_days/(len(test_df)-j),2)))
  return test_df, profit_dollars

j = 1
df_lag, shift = setup_lags(df,j)
x_train, y_train, x_test, y_test, train, test = seperate_data(df_lag, training_percentage, shift)
test += train['Adj Close'][-1]-test['Adj Close'][0]

# Linear Regression
print("Linear Regression")
lr_pred = LinearRegression_fnc(x_train, y_train, x_test, y_test)
lr_pred += train['Adj Close'][-1]-lr_pred[0]
test2, profit_dollars = CalcProfit(test, lr_pred,j)
plot(train, test, lr_pred, ticker, j, 'Linear Regression')  

# Artificial Neuarl Network 
print("ANN")
MLP_pred = ANN_func(x_train, y_train, x_test, y_test)
MLP_pred += train['Adj Close'][-1]-MLP_pred[0]
test2, profit_dollars = CalcProfit(test, MLP_pred, j)
plot(train, test, MLP_pred, ticker, j, 'Artificial Neuarl Network')

print(len(test), len(lr_pred), len(MLP_pred))
# exit(1)
lr_diffs = [abs(test['Adj Close'][i]-lr_pred[i]) for i in range(len(lr_pred))]
ann_diffs = [abs(test['Adj Close'][i]-MLP_pred[i]) for i in range(len(MLP_pred))]

_, axs = plt.subplots(2, 1)
axs[0].bar(list(range(len(lr_diffs))), lr_diffs, label='Linear regression method difference')       
plt.legend()
axs[1].bar(list(range(len(ann_diffs))), ann_diffs, label='Artificial Neural Network method difference')     
plt.legend()
plt.show()
