from flask import Flask, jsonify, request
from StockData import StockData

app = Flask(__name__)

class Data:
    def __init__(self, json):
        self.json = json

    def convertToJSON(self):
        return jsonify(self.json)


@app.route("/", methods=["POST"])
def ticker():
    # Send back 1MIN chart
    ticker = request.json["ticker"]
    dataframe = StockData("AAPL").handler()
    # analytics = Analytics(stock_data)
    # predicted = analytics.predict()
    # log = analytics.logs()

    return dataframe.to_json()

