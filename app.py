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
        self._json = {}

    def data(self):
        self._data = StockData(self.ticker)
        intervals = {
            '1D': self._data.get1dData,
            '5D': self._data.get5dData,
            '1M': self._data.get1moData,
            '3M': self._data.get3moData,
            '6M': self._data.get6moData,
            '1Y': self._data.get1YrData,
            'ALL': self._data.getAllData
        }
        threads = [th.Thread(target=self._call_back, args=(func, key)) for key, func in intervals.items()]
        for thread in threads: thread.start()
        for thread in threads: thread.join()

        return self._json

    def _call_back(self, func, key):
        try:
            self._json[key] = self.analytics.get(func())
        except:
            print('testttttttt')
            self._json[key] = None

@app.route("/one", methods=["POST"])
def ticker():
    # Send back 1MIN chart
    ticker = request.json["ticker"]
    dataframe = StockData(ticker).handler()
    analytics = Analyzer()
    predicted = analytics.get(dataframe)
    # analytics = Analytics(stock_data)
    # predicted = analytics.predict()
    # log = analytics.logs()

    return jsonify(predicted)

@app.route("/all", methods=["POST"])
def all_data():
    ticker = request.json["ticker"]
    newData = Data(ticker).data()
    return newData
