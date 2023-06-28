#Program to make decisions on the stock market based on data from SPY

import tensorflow as tf
from numpy import argmax
from pandas import read_csv

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


#Constants
OVERBOUGHT = 70 #relative strength index (RSI) over 70 is overbought
OVERSOLD = 30  #RSI under 30 is oversold
MAX_EPOCHS = 200

path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
df = read_csv(path, header=None)

X,y = df.values[:, :-1], df.values[:, -1]

X = X.astype('float32')

y = LabelEncoder().fit_transform(y)

model = tf.keras.Sequential()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

features = X_train.shape[1]

model.add(tf.keras.layers.Dense(10, activation = 'relu', kernel_initializer = 'he_normal', input_shape = (features,)))
model.add(tf.keras.layers.Dense(8, activation = 'relu', kernel_initializer = 'he_normal'))
model.add(tf.keras.layers.Dense(3, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = MAX_EPOCHS, batch_size = 32, verbose = 0)



loss,acc = model.evaluate(X_test, y_test, verbose = 0)
print('Accuracy: %.3f' % acc)
row = [5.1, 3.5, 1.4, 0.2]
yhat = model.predict([row])

print("Predicted: %s (class=%d)" % (yhat, argmax(yhat)))
"""
Some libraries to consider:
Apache Spark and Spark MLlib - for big data    
    
"""


#Strageries to consider
#1. Mean Reversion: regression to the mean.  Belief that prices will eventually revert back to the mean or average price. Usually based on daily return distribution.
#1.1 Pairs Trading: long one stock and short another.  Two stocks that are highly correlated but have diverged.  When the spread between the two stocks diverges, short the higher one and long the lower one.  When the spread converges, close the positions.
#1.2 Selling Options: selling options when implied violitility is high, and buying when it is low.  Belief that implied volatility will revert to the mean.


#2. Statistical Arbitrage: uses dozens of securities and looks at correlations. Used by professional traders.  Uses a basket of stocks that are highly correlated.  When the correlation breaks down, short the higher one and long the lower one. 

#3. Momentum: Try to profit based on price moves.
#3.1 Gap & Go: Opposite of mean reversion. When a stock gaps up, buy it and sell it when it gaps down. Good for earnings.

#4. Trend Following: Taking advantage of trends of a stock. Uses trend-following indicators like moving averages.  Buy when the price is above the moving average and sell when it is below.

#5. Market Making: Buy and sell at the bid and ask.  Make money on the spread.  Requires a lot of capital and is very risky.

#6. Sentiment Analysis: Uses news and social media to predict stock prices.  Uses natural language processing to analyze text.  










