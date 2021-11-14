import yfinance as yf

class StockData:
    def __init__(self, ticker, init=True):
        self.ticker = yf.Ticker(ticker)
        self.init = init
        #download ticker data, >3mo = 1d int, 6m = 1d, 1y = 1d, default=all, return adj close list, handle garbage tickers, get running parallel

    def handler(self):
        if self.init == True:
            data = self.get1dData()
            self.init = False
            return data
        else:
            return get5dData(), get1moData(), get3moData(), get6moData(), get1YrData(), getAllData()

    def get1dData(self):
        return yf.download(self.ticker.ticker, period='1d', interval='1m')

    def get5dData(self):
        return yf.download(self.ticker.ticker, period='5d', interval='1m')

    def get1moData(self):
        return yf.download(self.ticker.ticker, period='1mo', interval='5m')

    def get3moData(self):
        return yf.download(self.ticker.ticker, period='3mo', interval='1d')

    def get6moData(self):
        return yf.download(self.ticker.ticker, period='6mo', interval='1d')
        
    def get1YrData(self):
        return yf.download(self.ticker.ticker, period='1y', interval='1d')

    def getAllData(self):
        return yf.download(self.ticker.ticker, period = 'max', interval = '1d')

print(stockData('AAPL', True).handler())



