# import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.svm import SVR
import threading as th
import json
import talib

import Data, API

class Analyzer:
    def get(self, data):
        self.data = data
        # self.open = list(data['Open'])
        # self.high = list(data['High'])
        # self.low = list(data['Low'])
        # self.close = list(data['Close'])
        self.adj_close = list(data['Adj Close'])
        # self.volume = list(data['Volume'])
        # self._graph(self.adj_close, self.create_pridicted_data(self.adj_close)) 

        self.analytics = {
            'Data': self.adj_close,
            'Predicted': self.create_pridicted_data(self.adj_close),
            'Analytics note': '',
            'logs': []
        }

        return self.analytics

    def pattern_recignition(self, data):
        chunked_data = self._data_in_chunks(data)
        

    def create_pridicted_data(self, data):
        chunked_data, chunk_length = self._data_in_chunks(data)
        predictions = [self._predict(self._join(chunked_data[:i]), self._join(chunked_data[i+1:]), chunk_length) for i in range(len(chunked_data))]
        prediction = []
        for _prediction in predictions: prediction += _prediction
        return prediction+self.predict(data, chunk_length)

    def predict(self, data, interval):
        sample_size = len(data)
        indecies = [[index] for index in range(len(data))]
        RBF = SVR()
        RBF.fit(indecies, data)
        predicted = [RBF.predict([[index]])[0] for index in range(len(data), sample_size+int(sample_size*0.2))]
        pre = predicted[0]
        p_vals = []
        for i, p in enumerate(predicted):
            if i%interval == 0: pre = p
            p_vals.append(pre)
        predicted = p_vals
        return predicted

    def _predict(self, first_end, last_end, prediction_length):
        if first_end:
            first_end_indecies = [[index] for index in range(len(first_end))]
            first_end_regression = SVR()
            first_end_regression.fit(first_end_indecies, first_end)
            first_end_prediction = [first_end_regression.predict([[index+len(first_end)]])[0] for index in range(prediction_length)]    
        else:
            first_end_prediction = []
        
        if last_end:  
            last_end = last_end[::-1]
            last_end_indecies = [[index] for index in range(len(last_end))]
            last_end_regression = SVR()
            last_end_regression.fit(last_end_indecies, last_end)
            last_end_prediction = [last_end_regression.predict([[index+len(last_end)]])[0] for index in range(prediction_length)]
        else:
            last_end_prediction = []

        prediction = [(first+last)/2 for first, last in zip(first_end_prediction, last_end_prediction[::-1])]         
        return prediction

    def _data_in_chunks(self, data, interval_rate=0.01):
        interval = int(len(data)*interval_rate)
        chunks = []
        pre = 0
        for curr in range(interval, len(data), interval):
            chunks.append(data[pre: curr])
            pre = curr
        return chunks, interval

    def _join(self, data):
        values = []
        for value in data: values += value
        return values

    # def _graph(self, *args):
    #     number_of_graphs = len(args)
    #     # fig, axs = plt.subplots(number_of_graphs, 1)
    #     for index in range(number_of_graphs): 
    #         plt.plot(list(range(len(args[index]))), list(args[index]))
    #     # fig.tight_layout()
    #     plt.show()

if __name__ == '__main__':
    data = yf.download('AAPL', period='5d', interval='1m')
    App = Analyzer()
    App.get(data)