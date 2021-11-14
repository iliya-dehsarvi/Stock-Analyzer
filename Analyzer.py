import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.svm import SVR
import threading as th
import json
import talib

import Data, API

class Analyzer:
    class Analytics:
        def __init__(self):
            self.data = None
            self.predicted_data = None
            self.analytics_note = None
            self.logs = None

    def __init__(self, data):
        self.data = data
        self.open = list(data['Open'])
        self.high = list(data['High'])
        self.low = list(data['Low'])
        self.close = list(data['Close'])
        self.adj_close = list(data['Adj Close'])
        self.volume = list(data['Volume'])
        self.graph(self.adj_close, self.create_pridicted_data(self.adj_close), self.volume) 

    def graph(self, *args):
        number_of_graphs = len(args)
        fig, axs = plt.subplots(number_of_graphs, 1)
        for index in range(len(axs)): axs[index].plot(list(range(len(args[index]))), list(args[index]))
        fig.tight_layout()
        plt.show()

    def pattern_recignition(self): pass

    def create_pridicted_data(self, data):
        chunked_data, chunk_length = self._data_in_chunks(data)
        predictions = [self._predict(self._join(chunked_data[:i]), self._join(chunked_data[i+1:]), chunk_length) for i in range(len(chunked_data))]
        prediction = []
        for _prediction in predictions: prediction += _prediction
        return prediction

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

if __name__ == '__main__':
    data = yf.download('AAPL', period='5d', interval='1m')
    App = Analyzer(data)