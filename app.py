from flask import Flask, jsonify, request
from StockData import StockData
from Analyzer import Analyzer

app = Flask(__name__)

class Data:
    def __init__(self, json):
        self.json = json

    def convertToJSON(self):
        return jsonify(self.json)


@app.route("/one", methods=["POST"])
def ticker():
    # Send back 1MIN chart
    ticker = request.json["ticker"]
    dataframe = StockData("AAPL").handler()
    returnData = {
        'historical':list(dataframe['Adj Close'])
    }
    # analytics = Analytics(stock_data)
    # predicted = analytics.predict()
    # log = analytics.logs()

    return jsonify(returnData)


# import Threading as th

# threads = [th.Thread(target=call_back, args=(1,)]
# for thread in threads: thread.start()
# for thread in threads: thread.join()