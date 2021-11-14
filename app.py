from flask import Flask, jsonify, request
from StockData import StockData
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class Data:
    def __init__(self, json):
        self.json = json

    def convertToJSON(self):
        return jsonify(self.json)


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