from flask import Flask, jsonify, request
from StockData import StockData
from Analyzer import Analyzer
import threading as th
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class Data:
    def __init__(self, ticker):
        self.ticker = ticker
        self._data = {}
        self.analytics = Analyzer()

    def data(self):
        self._data = StockData(ticker)

        intervals = {
            '1d': d.get1dData,
            '5d': d.get5dData,
            '1mo': d.get1moData,
            '3mo': d.get3moData,
            '6mo': d.get6moData,
            '1yr': d.get1yrData,
            'all': d.getAllData
        }
        threads = [th.Thread(target=call_back, args=(func, key)) for key, func in intervals.items()]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        _json = {
            'ticker': ticker,
            'intervals': _data
        }

    def _call_back(self, func, key):
        self._data[key] = self.analytics.get(func())


@app.route("/one", methods=["POST"])
def ticker():
    # Send back 1MIN chart
    ticker = request.json["ticker"]
    dataframe = StockData(ticker).handler()
    returnData = {
        'historical':list(dataframe['Adj Close'])
    }
    # analytics = Analytics(stock_data)
    # predicted = analytics.predict()
    # log = analytics.logs()

    return jsonify(returnData)

@app.route("/two", methods=["POST"])
def all_data():
    ticker = request.json["ticker"]
    newData = Data(ticker).data()
    return newData
