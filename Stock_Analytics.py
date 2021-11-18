# # from scipy.sparse import data
# import yfinance as yf
# # import math
# # import matplotlib.pyplot as plt
# # from yfinance import ticker
# # import keras
# # import pandas as pd
# # import numpy as np
# # from keras.models import Sequential
# # from keras.layers import Dense
# # from keras.layers import LSTM
# # from keras.layers import Dropout
# # from keras.layers import *
# # from sklearn.preprocessing import MinMaxScaler
# # from sklearn.metrics import mean_squared_error
# # from sklearn.metrics import mean_absolute_error
# # from sklearn.model_selection import train_test_split
# # from keras.callbacks import EarlyStopping


# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import matplotlib
# # from sklearn.preprocessing import MinMaxScaler
# # from keras.layers import LSTM, Dense, Dropout
# # from sklearn.model_selection import TimeSeriesSplit
# # from sklearn.metrics import mean_squared_error, r2_score
# # import matplotlib.dates as mandates
# # from sklearn.preprocessing import MinMaxScaler
# # from sklearn import linear_model
# # from keras.models import Sequential
# # from keras.layers import Dense
# # import keras.backend as K
# # from keras.callbacks import EarlyStopping
# # from keras.optimisers import Adam
# # from keras.models import load_model
# # from keras.layers import LSTM
# # from keras.utils.vis_utils import plot_model

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.pylab import rcParams
# rcParams['figure.figsize'] = 20, 10
# from keras.models import Sequential
# from keras.layers import LSTM, Dropout, Dense

# from sklearn.preprocessing import MinMaxScaler

# class Analytics:
#     def __init__(self, ticker='AAPL', period='5d', interval='1m'):
#         self.ticker = ticker
#         self.df = self._download(ticker, period, interval)
#         # dates = [str(date).split()[0] for date in list(self.df.index)]
#         # self.df['Date'] = dates

#         # self.df['Date'] = pd.to_datetime(self.df.Date, format='%Y-%m-%d')
#         # self.df.index = self.df['Date']

#         # data = self.df.sort_index(ascending=True, axis=0)
#         # new_dataset = pd.DataFrame(index=range(0, len(self.df)), columns=['Date','Close'])
#         # for i in range(0, len(data)):
#         #     new_dataset['Date'][i] = data['Date'][i]
#         #     new_dataset['Close'][i] = data['Close'][i]

        
#         # print(dates)
#         # print(self.df.head(5))
#         # exit(1)
#         self._predict(self.df['Adj Close'])

#     def _download(self, ticker, period, interval):
#         try: return yf.download(ticker)#, period=period, interval=interval)
#         except:
#             print('Something went wrong will downloading the data...')
#             exit(1)

#     def _predict(self, df):
#         # print(df)
#         # exit(1)
#         training_length = int(len(df)*0.8)
#         scaler = MinMaxScaler(feature_range=(0,1))
#         final_dataset = list(df.values)
#         train_data = final_dataset[:training_length]
#         valid_data = final_dataset[training_length:]
#         # df.index = df.Date
#         # df.drop('Date', axis=1, inplace=True)
#         scaler=MinMaxScaler(feature_range=(0, 1))
#         # final_dataset = final_dataset.reshape(1, -1)
#         # exit(1)
#         scaled_data=scaler.fit_transform(final_dataset)
#         # print(scaled_data.shape)
#         # exit(1)
#         x_train_data, y_train_data = [], []
#         for i in range(60, len(train_data)):
#             x_train_data.append(scaled_data[i-60:i])
#             y_train_data.append(scaled_data[i])
            
#         x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
#         x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

#         lstm_model=Sequential()
#         lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
#         lstm_model.add(LSTM(units=50))
#         lstm_model.add(Dense(1))
#         inputs_data=df[len(df)-len(valid_data)-60:].values
#         inputs_data=inputs_data.reshape(-1,1)
#         inputs_data=scaler.transform(inputs_data)
#         lstm_model.compile(loss='mean_squared_error',optimizer='adam')
#         lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose=2)

#         X_test=[]
#         for i in range(60,inputs_data.shape[0]):
#             X_test.append(inputs_data[i-60:i,0])
#         X_test=np.array(X_test)
#         X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
#         predicted_closing_price=lstm_model.predict(X_test)
#         predicted_closing_price=scaler.inverse_transform(predicted_closing_price)
        
#         lstm_model.save("saved_model.h5")

#         train_data=df[:987]
#         valid_data=df[987:]
#         valid_data['Predictions']=predicted_closing_price
#         plt.plot(train_data['Close'])
#         plt.plot(valid_data[['Close','Predictions']])



#         '''
#         output_var = pd.DataFrame(df['Adj Close'])
#         features = ['Open', 'High', 'Low', 'Volume']
#         scaler = MinMaxScaler()
#         feature_transform = scaler.fit_transform(df[features])
#         feature_transform= pd.DataFrame(columns=features, data=feature_transform, index=df.index)
#         feature_transform.head()

#         timesplit= TimeSeriesSplit(n_splits=10)
#         for train_index, test_index in timesplit.split(feature_transform):
#             X_train, X_test = feature_transform[:len(train_index)], feature_transform[len(train_index): (len(train_index)+len(test_index))]
#             y_train, y_test = output_var[:len(train_index)].values.ravel(), output_var[len(train_index): (len(train_index)+len(test_index))].values.ravel()
        
#         trainX = np.array(X_train)
#         testX = np.array(X_test)
#         X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
#         X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])

#         lstm = Sequential()
#         lstm.add(LSTM(32, input_shape=(1, trainX.shape[1]), activation='relu', return_sequences=False))
#         lstm.add(Dense(1))
#         lstm.compile(loss='mean_squared_error', optimizer='adam')
#         plot_model(lstm, show_shapes=True, show_layer_names=True)
        
#         y_pred = lstm.predict(X_test)

#         plt.plot(y_test, label='True Value')
#         plt.plot(y_pred, label='LSTM Value')
#         plt.title('Prediction by LSTM')
#         plt.xlabel('Time Scale')
#         plt.ylabel('Scaled USD')
#         plt.legend()
#         plt.show()
#         '''

#         '''
#         training_length = int(len(df)*0.8)
#         time_stamps = int(training_length*0.1)
#         training_set = df.iloc[:training_length, 1:2].values
#         test_set = df.iloc[training_length:, 1:2].values

#         sc = MinMaxScaler(feature_range=(0,1))
#         training_set_scaled = sc.fit_transform(training_set)

#         x_train = []
#         y_train = []
#         for i in range(time_stamps, training_length):
#             x_train.append(training_set_scaled[i-time_stamps:i, 0])
#             y_train.append(training_set_scaled[i:i, 0])
#         x_train, y_train = np.array(x_train), np.array(y_train)
#         x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#         model = Sequential()
#         model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
#         model.add(Dropout(0.2))
#         model.add(LSTM(units = 50, return_sequences = True))
#         model.add(Dropout(0.2))
#         model.add(LSTM(units = 50, return_sequences = True))
#         model.add(Dropout(0.2))
#         model.add(LSTM(units = 50))
#         model.add(Dropout(0.2))
#         model.add(Dense(units = 1))
#         model.compile(optimizer = 'adam', loss = 'mean_squared_error')
#         model.fit(x_train, y_train, epochs = 100, batch_size = 32)

#         dataset_train = df.iloc[:training_length, 1:2]
#         dataset_test = df.iloc[training_length:, 1:2]
#         dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
#         inputs = dataset_total[len(dataset_total) - len(dataset_test) - time_stamps:].values
#         inputs = inputs.reshape(-1,1)
#         inputs = sc.transform(inputs)
#         x_test = [inputs[i-time_stamps:i, 0] for i in range(time_stamps, training_length)]
#         x_test = np.array(x_test)
#         x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#         predicted_stock_price = model.predict(x_test)
#         predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        
#         plt.plot(list(range(len(dataset_test.values))), dataset_test.values, color = 'red', label = 'Real '+self.ticker+' Stock Price')
#         plt.plot(list(range(len(predicted_stock_price))), predicted_stock_price, color = 'blue', label = 'Predicted '+self.ticker+' Stock Price')
#         plt.xticks(np.arange(x_test.shape[0], x_test.shape[1], x_test.shape[2]))
#         plt.title(self.ticker+' Stock Price Prediction')
#         plt.xlabel('Time')
#         plt.ylabel(self.ticker+' Stock Price')
#         plt.legend()
#         plt.show()
#         '''

# if __name__ == '__main__':
#     App = Analytics()