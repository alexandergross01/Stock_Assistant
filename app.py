from FinBERT_Training import FinBERT
from datetime import datetime
import streamlit as st
import pandas as pd
import yfinance as yf
from Models import Algo_Models, GRU_Model
import torch

class Stock_Assistant():
    def __init__(self, tickers_dict):
        st.set_page_config(layout="wide")
        st.title("Tech Stocks Trading Assistant")

        self.left_column, self.right_column = st.columns(2)
        self.tickers = tickers_dict
    
    def run(self):
        with self.left_column:
            st.subheader("Algorithm Analysis Methods")
            self.option_name = st.selectbox("Select Stock:", self.tickers.keys())
            self.selection = self.tickers[self.option_name]
            self.curr_time = datetime.now()
            'Selected: ', self.option_name, '(',self.selection, ')',
            'Last Execution:', self.curr_time

            models = Algo_Models(self.selection)
            with st.spinner('Loading stock data...'):
                technical_analysis_methods_outputs = {
                    'Technical Analysis Method': [
                        'Bollinger Bands (20 days & 2 stand. deviations)', 
                        'Bollinger Bands (10 days & 1.5 stand. deviations)', 
                        'Bollinger Bands (50 days & 3 stand. deviations)', 
                        'Moving Average Convergence Divergence (MACD)'
                        ],
                        'Outlook': [
                            models.Bollinger_Band(20, 2),         
                            models.Bollinger_Band(10, 1.5), 
                            models.Bollinger_Band(50, 3), 
                            models.MACD()
                            ],
                        'Timeframe of Method': [
                            "Medium-term",         
                            "Short-terrm", 
                            "Long-term", 
                            "Short-term"
                            ]
                        }

                self.df = pd.DataFrame(technical_analysis_methods_outputs)
            
            def color_survived(val):
                color = ""
                if (val=="Sell" or val=="Downtrend and sell signal" or val=="Downtrend and no signal"):
                    color="#EE3B3B"
                elif (val=="Buy" or val=="Uptrend and buy signal" or val=="Uptrend and no signal"):
                    color="#3D9140"
                else:
                    color="#CD950C"
                return f'background-color: {color}'

            st.table(self.df.sort_values(['Timeframe of Method'], ascending=False).
                    reset_index(drop=True).style.applymap(color_survived, subset=['Outlook']))

        with self.right_column:
            st.subheader("FinBERT-based Sentiment Analysis")

            with st.spinner("Connecting with www.marketwatch.com..."):
                dict = models.finbert_headlines_sentiment()
                st.plotly_chart(dict["fig"])
                "Current sentiment:", dict["current_sentiment"], "%"

            st.subheader("LSTM-based 7-day stock price prediction model")

            with st.spinner("Compiling GRU model.."):
                nn = GRU_Model()
                nn.train(self.selection, 25, 50)

                with torch.no_grad():
                    nn.lookahead(7)
                st.pyplot(fig=nn.plotting_prediction())

if __name__ == "__main__":
    all_tickers = {
                "Apple":"AAPL", 
                "Microsoft":"MSFT", 
                "Nvidia":"NVDA", 
                "Paypal":"PYPL",
                "Amazon":"AMZN",
                "Spotify":"SPOT",
                #"Twitter":"TWTR",
                "AirBnB":"ABNB",
                "Uber":"UBER",
                "Schwab": "SCHD",
                "Google":"GOOG"
                }
    Assistant = Stock_Assistant(all_tickers)
    Assistant.run()
