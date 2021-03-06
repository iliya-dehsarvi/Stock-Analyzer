import yfinance as yf

class StockData:
    def __init__(self, ticker, init=True):
        self.ticker = ticker
        self.init = init
        #download ticker data, >3mo = 1d int, 6m = 1d, 1y = 1d, default=all, return adj close list, handle garbage tickers, get running parallel

    def handler(self):
        if self.init == True:
            data = self.get1dData()
            self.init = False
            return data
        else:
            return self.get5dData(), self.get1moData(), self.get3moData(), self.get6moData(), self.get1YrData(), self.getAllData()

    def get1dData(self):
        return yf.download(self.ticker, period='1d', interval='1m')

    def get5dData(self):
        return yf.download(self.ticker, period='5d', interval='1m')

    def get1moData(self):
        return yf.download(self.ticker, period='1mo')

    def get3moData(self):
        return yf.download(self.ticker, period='3mo')

    def get6moData(self):
        return yf.download(self.ticker, period='6mo')
        
    def get1YrData(self):
        return yf.download(self.ticker, period='1y')

    def getAllData(self):
        return yf.download(self.ticker, period = 'max')




