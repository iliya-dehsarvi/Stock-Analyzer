import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.svm import SVR
import json

import Data, API

class Analytics:
    def __init__(self):
        self.data = None
        self.predicted_data = None
        self.logs = None

class Analyzer:
    def __init__(self, data):
        self.data = data
        self.open = data['Open']
        self.high = data['High']
        self.low = data['Low']
        self.close = data['Close']
        self.adj_close = data['Adj Close']
        self.volume = data['Volume']
        self.graph(self.open, self.close, self.volume) 

    def graph(self, *args):
        number_of_graphs = len(args)
        fig, axs = plt.subplots(number_of_graphs, 1)
        for index in range(len(axs)): axs[index].plot(list(range(len(args[index]))), list(args[index]))
        fig.tight_layout()
        plt.show()

if __name__ == '__main__':
    data = yf.download('AAPL', period='5d', interval='1m')
    with open('/Users/iliyadehsarvi/Documents/GitHub/Stock-Analyzer/api.json', 'w') as api:
        api.write(data.to_json())
    # App = Analyzer(data)