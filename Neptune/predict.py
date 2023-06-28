import neptune
import numpy as np
import matplotlib.pyplot as plt    
import urllib.request, json
import os
import pandas as pd
import datetime as dt


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential, Model
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
"""
START OF COPY

"""



data_source = 'alphavantage'


if data_source == 'alphavantage':
    # ====================== Loading Data from Alpha Vantage ==================================
    api_key = 'XRG0MVBAVSSOMQVH'
    # stock ticker symbol
    ticker = 'AAPL' 
    
    # JSON file with all the stock prices data 
    url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)

    # Save data to this file
    fileName = 'stock_market_data-%s.csv'%ticker

    ### get the low, high, close, and open prices 
    if not os.path.exists(fileName):
        with urllib.request.urlopen(url_string) as url:
            data = json.loads(url.read().decode())
            # pull stock market data
            data = data['Time Series (Daily)']
            df = pd.DataFrame(columns=['Date','Low','High','Close','Open'])
            for key,val in data.items():
                date = dt.datetime.strptime(key, '%Y-%m-%d')
                data_row = [date.date(),float(val['3. low']),float(val['2. high']),
                            float(val['4. close']),float(val['1. open'])]
                df.loc[-1,:] = data_row
                df.index = df.index + 1
        df.to_csv(fileName)

    else:
        print('Loading data from local')
        df = pd.read_csv(fileName)
    
# Sort DataFrame by date
stockprices = df.sort_values('Date')



""" 
END OF COPY
"""






test_ratio = 0.2
train_ratio = 1 - test_ratio

train_size = int(train_ratio * len(stockprices))
test_size = int(test_ratio * len(stockprices))
print(f"Train size: {train_size}")
print(f"Test size: {test_size}")

train = stockprices[:train_size][["Date", "Close"]]
test = stockprices[train_size:][["Date", "Close"]]


# Split the values of the time-series into training set X and output Y 
def extract_seqX_outcomeY(data, N, offset):
    X, y = [], []
    
    """
    Args:
        data: Dataset of stock prices
        N: Window size, number of days to look back
        offset: Position to start the split
    """
    
    for i in range(offset, len(data)):
        X.append(data[i-N:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Calculate RMSE and MAPE
def calculate_rmse(y_true, y_pred):
    """
    Root Mean Squared Error
    """
    return np.sqrt(np.mean((y_true - y_pred)**2))
    
def calculate_mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


#Evaluate the metrics
def calculate_perf_metrics(var):
    #RMSE
    rmse = calculate_rmse(
        np.array(stockprices[train_size:]['Close']),
        np.array(stockprices[train_size:][var])
    )
    mape = calculate_mape(
        np.array(stockprices[train_size:]['Close']),
        np.array(stockprices[train_size:][var]),
    )
    #Log results to Neptune    
    run["RMSE"] = rmse
    run["MAPE"] = mape
    
    return rmse, mape



# Plot trend of stock prices and log plot to Neptune
def plot_stock_trend(var, cur_title, stockprices = stockprices):
    ax = stockprices[["Close", var, "200day"]].plot(figsize=(20,10))
    plt.grid(False)
    plt.title(cur_title)
    plt.axis('tight')
    plt.ylabel('Stock Price ($)')
    
    #log to Neptune
    run["Plot of Stock Predictions"].upload(
        neptune.types.File.as_image(ax.get_figure())
    )



# def 


#end of functions







""" 
RUN TIME PORTION
"""

window_size = 50


run = neptune.init_run(
    project="rileycyeoman/StockPrediction",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3MGEzY2RjZi0yNDkyLTQwMDktYjIzNC00MGRlNGJkMmQxYzQifQ==",
    description="Predction of stock prices using LSTM",
)  # your credentials

window_var = f"{window_size}day"
stockprices[window_var] = stockprices['Close'].rolling(window=window_size).mean()

# Include a 200-day SMA for reference
stockprices['200day'] = stockprices['Close'].rolling(200).mean()

# Plot the performance metrics
plot_stock_trend(var = window_var, cur_title="Simple Moving Average")
rmse_sma, mape_sma = calculate_perf_metrics(var = window_var)
run.stop()

 
# params = {"learning_rate": 0.001, "optimizer": "Adam"}
# run["parameters"] = params

# for epoch in range(10):
#     run["train/loss"].append(0.9 ** epoch)

# run["eval/f1_score"] = 0.66

# run.stop()