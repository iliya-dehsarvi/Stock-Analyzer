import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.svm import SVR
import threading as th
import json
import talib as ta
import matplotlib.pyplot as plt

class Analyzer:
    def get(self, data):
        self.data = data
        self.logs = ''
        self.adj_close = list(data['Adj Close'])
        self.volume = list(data['Volume'])
        self.pattern_recignition(self.data)
        cross_validation_prediction = self.create_pridicted_data(self.adj_close)
        self.analytics = {
            'historical': self.adj_close,
            'cross-validation-prediction': cross_validation_prediction,
            # 'predicted': predicted,
            'notes': '',
            'logs': self.logs 
        }

        print(self.logs)
        if len(self.analytics['cross-validation-prediction']) < len(self.adj_close): first, second = cross_validation_prediction, self.adj_close
        else: second, first = cross_validation_prediction, self.adj_close
        diffs = [(abs(first[i]-second[i])*100)/first[i] for i in range(len(first))]
        cross_validation_prediction_volume = self.create_pridicted_data(diffs)

        # diffs += [0]*(len(self.analytics['predicted'])-len(self.adj_close))
        # self.volume += [0]*(len(self.analytics['predicted'])-len(self.adj_close))
        self._graph(self.adj_close, cross_validation_prediction, cross_validation_prediction_volume, self.volume, diffs)
        return self.analytics

    def pattern_recignition(self, DF):
        patterns = {
            'CDLHAMMER' : ta.CDLHAMMER(DF['Open'], DF['High'], DF['Low'], DF['Close']),
            'CDLENGULFING' : ta.CDLENGULFING(DF['Open'], DF['High'], DF['Low'], DF['Close']),
            'CDL3BLACKCROWS' : ta.CDL3BLACKCROWS(DF['Open'], DF['High'], DF['Low'], DF['Close']),
            'CDL3LINESTRIKE' : ta.CDL3LINESTRIKE(DF['Open'], DF['High'], DF['Low'], DF['Close']),
            'CDLSTICKSANDWICH' : ta.CDLSTICKSANDWICH(DF['Open'], DF['High'], DF['Low'], DF['Close']),
            'CDL3WHITESOLDIERS' : ta.CDL3WHITESOLDIERS(DF['Open'], DF['High'], DF['Low'], DF['Close'])
        }

        for name, pattern in patterns.items():
            print(name)
            print(pattern[pattern == 100].dropna())

    def create_pridicted_data(self, data):
        chunked_data, chunk_length = self._data_in_chunks(data)
        predictions = [self._predict(self._join(chunked_data[:i]), self._join(chunked_data[i+1:]), chunk_length) for i in range(len(chunked_data))]
        prediction = []
        for _prediction in predictions: prediction += _prediction
        return prediction+self.predict(prediction)

    def predict(self, data):
        sample_size = len(data)
        indecies = [[index] for index in range(len(data))]
        RBF = SVR()
        RBF.fit(indecies, data)
        predicted = [RBF.predict([[index]])[0] for index in range(len(data), sample_size+int(sample_size*0.02))]
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
        chunks = [data[x:x+interval] for x in range(0, len(data), interval)]
        chunks += [data[len(data)%interval:]]
        # chunks = []
        # pre = 0
        # for curr in range(interval, len(data), interval):
        #     chunks.append(data[pre: curr])
        #     pre = curr
        return chunks, interval

    def _join(self, data):
        values = []
        for value in data: values += value
        return values

    def _graph(self, data, cross_validation_prediction, cross_validation_prediction_volume, volume, diffs):
        # predicted = [None]*len(cross_validation_prediction)+predicted
        fig, axs = plt.subplots(4, 1)
        for graph in (data, cross_validation_prediction): axs[0].plot(list(range(len(graph))), graph)
        axs[1].bar(list(range(len(cross_validation_prediction_volume))), cross_validation_prediction_volume)
        axs[2].bar(list(range(len(volume))), volume)
        axs[3].bar(list(range(len(diffs))), diffs)
        fig.tight_layout()
        plt.show()

if __name__ == '__main__':
    data = yf.download('AAPL', period='1d', interval='1m')
    App = Analyzer()
    App.get(data)