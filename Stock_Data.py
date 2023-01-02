from configparser import ParsingError
from logging import raiseExceptions
import yfinance as yf
import requests
import pandas as pd
from bs4 import BeautifulSoup

class Stock_Data():
    
    def __init__(self, ticker : str, nn_train = False) -> None:
        self.market_watch_link_part1 = 'https://www.marketwatch.com/investing/stock/'
        self.market_watch_link_part2 = '?mod=search_symbol'
      
        self.ticker= ticker
       
        self.market_watch_url = self.market_watch_link_part1 + \
            self.ticker +  self.market_watch_link_part2
        
        self.market_status = None
        self.price = None
        self.headlines = None

        if(not nn_train):
            two_yr_daily_data = yf.download(tickers = self.ticker,
                              period = "2y",
                              interval = '1d')

            self.yf_data= pd.DataFrame(two_yr_daily_data)
        else:
            daily_data = yf.download(tickers = self.ticker,
                              period = "max",
                              interval = '1d')

            self.yf_data= pd.DataFrame(daily_data)


    def load_market_status(self) -> None:
        """Gets status of exchange that the desired ticker is listed in
            Possible return values are: After Hours, Open, Market Closed"""
        self.webpage = requests.get(self.market_watch_url)
        self.soup = BeautifulSoup(self.webpage.text, "lxml")

        
        if self.soup is None:
            raise ParsingError("HTML code of MarketWatch website was not scraped and current status can not be found" + \
                "\nPlease check your ticker or if the following website exists:\n" + self.market_watch_url )
        else:
            self.market_status = self.soup.find("div", class_="status").text
    
    def load_price(self):
        #breakpoint()
        if(self.market_status == None):
            raise ParsingError("HTML code for MarketWatch website has not been loaded, please run" +\
                " get_market_status() function to update")
        else:
            self.price = float(self.soup.find("bg-quote", class_="value").text.replace(',',''))
    
    def parse_articles(self):
        results = self.soup.find("div", class_="tab__pane is-active j-tabPane")
        articles = results.find_all("a", class_="link")
        headerList = ["ticker", "headline"]
        rows = []
        counter = 1

        for art in articles:
            if counter <= 17:
                title = art.text.strip()
                if title is None:
                        break
                rows.append([self.ticker, title])
                counter = counter + 1

        self.headlines = pd.DataFrame(rows, columns=headerList)
