import Stock_Data
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import datetime as dt
from FinBERT_Training import FinBERT
import plotly.graph_objs as go  
import plotly.io as pio
pio.templates
pd.options.plotting.backend = "plotly"

class Algo_Models():
    def __init__(self, ticker) -> None:
        self.ticker = ticker
        self.stock_data = Stock_Data.Stock_Data(ticker)
        self.stock_data.load_market_status()
        self.stock_data.load_price()
        self.data = self.stock_data.yf_data

    def band_calc(self, window_len: int, std_multiplyer: float) -> tuple:
        

        sma = self.data.rolling(window=window_len).mean().dropna()
        rstd = self.data.rolling(window=window_len).std().dropna()

        upper = sma + std_multiplyer*rstd
        lower = sma - std_multiplyer*rstd
        upper = upper.rename(columns={'Close':'Upper'})
        lower = lower.rename(columns={'Close':'Lower'})
        data = self.data
        bands =  data.join(upper['Upper']).join(lower['Lower'])
        bands = bands.dropna()
        return sma, bands

    def band_decide(self, price, mean, recent_data):
        if price < mean and price >= recent_data['Lower']:
            return 'Buy'
        elif price > mean and price <= recent_data['Upper']:
            return 'Sell'
        elif price < recent_data['Lower'] or price > recent_data['Upper']:
            return 'Unusual Event'
        else:
            return 'Hold'

    def Bollinger_Band(self, window: int, std_multiplier: float) -> str:
        price = self.stock_data.price
        sma, bands = self.band_calc(window,std_multiplier)

        mean = sma.iloc[-1,:]['Close']
        recent_data = bands.iloc[-1,:]
 
        return self.band_decide(price, mean, recent_data)

    def MACD(self) -> str:
        ema12d = self.data['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
        ma26d = self.data['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
        macd = ema12d - ma26d
        ma_9d_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        
        diff = macd - ma_9d_signal
        diff.dropna()
        if diff.iloc[-1] < 0:
            if diff.iloc[-2] >= 0:
                return "Downtrend and sell signal"
            else:
                return "Downtrend and no signal"
        else:
            if diff.iloc[-2] <= 0:
                return "Uptrend and buy signal"
            else:
                return "Uptrend and no signal"

    def finbert_headlines_sentiment(self): 
        self.stock_data.parse_articles()
        articles_df = self.stock_data.headlines
        articles_list = articles_df["headline"].tolist()
        
        print(articles_list)
        model = FinBERT()
        outputs_list = model.input(articles_list)
        print(outputs_list)
        sentiments = []

        for item in outputs_list:
            sentiments.append(item["label"])
        #breakpoint()
        sentiments_df = pd.DataFrame(sentiments)
        sentiments_df.rename(columns = {0:'sentiment'}, inplace = True)

        sentiments_df["sentiment"] = sentiments_df["sentiment"].apply(lambda x: 100 if x.lower() == "positive" else -100 if x.lower()=="negative" else 0)            
        sentiments_df["roll_avg"] = round(sentiments_df["sentiment"].rolling(5, min_periods = 1).mean(), 2)
        sentiments_df = sentiments_df.tail(12).reset_index()

        pd.options.plotting.backend = "plotly"

        fig = sentiments_df["roll_avg"].plot(title="Sentiment Analysis of the last 12 www.marketwatch.com articles about " + self.ticker, 
        
        template="plotly_dark",
        labels=dict(index="17 most recent article headlines", value="sentiment  score (rolling avg. of window size 5)"))
        fig.update_traces(line=dict(color="#3D9140", width=3))
        fig.update_layout(yaxis_range=[-100,100])
        fig.update_layout(xaxis_range=[0,12])
        fig.update_layout(showlegend=False)
        fig.add_hline(y=0, line_width=1.5, line_color="black")
       
        current_sentiment = sentiments_df["roll_avg"].tail(1).values[0]

        return {'fig': fig, 'current_sentiment': current_sentiment}

class GRU_Model(nn.Module):
    def __init__(self, input_dim = 1, hidden_dim = 32, num_layers = 2, output_dim = 1):
        super(GRU_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.GRU = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.GRU(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out

    def generate_data(self, ticker, days_timeframe) -> None:
        self.ticker = ticker
        self.stock_data = Stock_Data.Stock_Data(ticker, nn_train= False)
        self.stock_data.load_market_status()
        self.stock_data.load_price()
        self.data = self.stock_data.yf_data
        prices = self.data['Close']

        self.scalar = MinMaxScaler(feature_range=(-1,1))
        prices = self.scalar.fit_transform(prices.values.reshape(-1,1))

        self.raw_prices = prices
        data = []

        for idx in range(len(self.raw_prices) - days_timeframe):
            data.append(self.raw_prices[idx : idx+days_timeframe])
        
        data = np.array(data)
        test_size = int(np.round(0.2*data.shape[0]))
        train_size = data.shape[0] - (test_size)

        self.x_train = data[:train_size,:-1,:]
        self.y_train = data[:train_size,-1,:]
    
        self.x_test = data[train_size:,:-1]
        self.y_test = data[train_size:,-1,:]

        print('x_train.shape = ',self.x_train.shape)
        print('y_train.shape = ',self.y_train.shape)
        print('x_test.shape = ',self.x_test.shape)
        print('y_test.shape = ',self.y_test.shape)

        self.x_train = torch.from_numpy(self.x_train).type(torch.Tensor)
        self.x_test = torch.from_numpy(self.x_test).type(torch.Tensor)
        self.y_train = torch.from_numpy(self.y_train).type(torch.Tensor)
        self.y_test = torch.from_numpy(self.y_test).type(torch.Tensor)

    def train(self, ticker, timeframe = 20, num_epochs = 100):
        self.hist = np.zeros(num_epochs)
        self.timeframe = timeframe
        gru = []
        self.generate_data(ticker, timeframe)
        criterion = torch.nn.MSELoss(reduction='mean')
        optimiser = torch.optim.Adam(self.parameters(), lr=0.01)
        for epoch in range(num_epochs):
            pred = self(self.x_train)
            loss = criterion(pred, self.y_train)
            print("Epoch: ", epoch, "MSE: ", loss.item())
            self.hist[epoch] = loss.item()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        
        torch.save(self.state_dict(), 'GRU_Test.pth')
        
        self.predict = pd.DataFrame(self.scalar.inverse_transform(pred.detach().numpy()))
        self.original = pd.DataFrame(self.scalar.inverse_transform(self.y_train.detach().numpy()))


    def plot_train_res(self):   
        sns.set_style("white") 
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.2, wspace=0.2)

        plt.subplot(1, 2, 1)
        ax = sns.lineplot(x = self.original.index, y = self.original[0], label="Data", color='royalblue')
        ax = sns.lineplot(x = self.predict.index, y = self.predict[0], label="Training Prediction (GRU)", color='tomato')
        ax.set_title('Stock price', size = 14, fontweight='bold')
        ax.set_xlabel("Days", size = 14)
        ax.set_ylabel("Cost (USD)", size = 14)
        ax.set_xticklabels('', size=10)


        plt.subplot(1, 2, 2)
        ax = sns.lineplot(data=self.hist, color='royalblue')
        ax.set_xlabel("Epoch", size = 14)
        ax.set_ylabel("Loss", size = 14)
        ax.set_title("Training Loss", size = 14, fontweight='bold')
        fig.set_figheight(6)
        fig.set_figwidth(16)
        #plt.show()

    def lookahead(self, days_ahead):
        pr = [self.raw_prices[-1][0]]
        prediction = self.raw_prices[-self.timeframe:]
        for days in range(days_ahead):
            pred = torch.from_numpy(prediction[-self.timeframe:]).type(torch.Tensor)
            pred = pred.reshape((1,self.timeframe,1))
            out = self(pred[-self.timeframe:])[0][0]
            #breakpoint()
            prediction = np.append(prediction, [out.tolist()])
            pr.append(out.tolist())
        self.prediction_outlook = prediction
        last_date = self.data.index[-1]
        self.prediction_dates = pd.date_range(last_date, periods=days_ahead+1)
        self.prediction_outlook = self.scalar.inverse_transform(self.prediction_outlook.reshape(-1,1))

    def plotting_prediction(self):
        sns.set_style("white") 
        plt.figure(figsize=(50,20))
        actual_dates = self.data
        actual_dates.reset_index(inplace=True)
        
        actual_dates['Date'] = pd.to_datetime(actual_dates['Date'])
        #plt.plot(actual_dates['Date'], actual_dates['Close'])
      
        locator = mdates.MonthLocator()
        X = plt.gca().xaxis
        X.set_major_locator(locator)

        pred_dates = pd.DataFrame({"Date":self.prediction_dates.to_pydatetime()})
        pred_dates['Date'] = pd.to_datetime(pred_dates['Date'])
        #pred_dates['Date'] = pred_dates['Date'].dt.strftime('%Y/%m/%d')

        #plt.plot(pred_dates['Date'], self.prediction_outlook[-8:], color='r')
        plot_1 = go.Scatter(
            x = actual_dates['Date'].iloc[-120:],
            y = actual_dates['Close'].iloc[-120:],
            mode = 'lines',
            name = 'Historical Data (2 years)',
            line=dict(width=1,color='#3D9140'))
        plot_2 = go.Scatter(
            x = pred_dates['Date'],
            y = self.prediction_outlook[-8:].squeeze(),
            mode = 'lines',
            name = '7-day Prediction',
            line=dict(width=1,color="#EE3B3B"))
        plot_3 = go.Scatter(
            x = actual_dates['Date'].iloc[-1:],
            y = actual_dates['Close'].iloc[-1:],
            mode = 'markers',
            name = 'Latest Actual Closing Price',
            line=dict(width=1))
    
        # adding bollinger bands to plot
        m = Algo_Models(self.ticker)
        _, data = m.band_calc(20, 2)
        plot_4 = go.Scatter(
            x = actual_dates['Date'].iloc[-120:],
            y = data['Upper'].iloc[-120:],
            mode = 'lines',
            name = 'Upper Band',
            line=dict(width=1,color='#8A2525'))
        plot_5 = go.Scatter(
            x = actual_dates['Date'].iloc[-120:],
            y = data['Lower'].iloc[-120:],
            mode = 'lines',
            name = 'Lower Band',
            line=dict(width=1,color='#0026FF'))
        
        layout = go.Layout(
            title = 'Next 7 days stock price prediction of ' + str(self.ticker),
            xaxis = {'title' : "Date"},
            yaxis = {'title' : "Price ($)"}
        )
        #breakpoint()
        fig = go.Figure(data=[plot_1, plot_2,plot_3,plot_4, plot_5], layout=layout)
        fig.update_layout(template='plotly_white',autosize=True)
        fig.update_layout(legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                          ))
        # fig.update_layout(legend=dict(
        #     orientation="h",
        #     yanchor="bottom",
        #     y=1.02,
        #     xanchor="right",
        #     x=1),
        #     annotations = [dict(x=0.5,
        #                         y=0, 
        #                         xref='paper',
        #                         yref='paper',
        #                         text="Current In Sample R- Squared : " + str(r_squared_score*100) + " % \n",
        #                         showarrow = False)],
        #     xaxis=dict(showgrid=False),
        #     yaxis=dict(showgrid=False)
            

        #                 )
        # fig.add_annotation(x=0.5, 
        #                    y=0.05,
        #                    xref='paper',
        #                    yref='paper',
        #                    text="Current In Sample Root Mean Square Error : " + str(round(rmse,2)) + " % ",
        #                    showarrow=False)
        #fig.show()
        return fig


if __name__ == '__main__':
    m = Algo_Models('F')
    print(m.Bollinger_Band(20,2))
    print(m.Bollinger_Band(10,1.5))
    print(m.Bollinger_Band(50, 3))
    print(m.MACD())
    
    m.finbert_headlines_sentiment()
    s = GRU_Model(1,32,2,1)
    s.train('AMZN', 20, 100)
    #s.plot_train_res()

    with torch.no_grad():
        s.lookahead(7)
    
    s.plotting_prediction()
    









