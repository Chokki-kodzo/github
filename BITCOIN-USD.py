#!/usr/bin/env python
# coding: utf-8

# #                         *Prediction of Bitcoin-USD Using ARIMA and LSTM* 
# 
Bitcoin has a rich history despite its young age. It is the first cryptocurrency in the world and the most widespread. Bitcoin is based on a technology called blockchain that can record transactions between two parties in a very efficient and verifiable way without the need for a financial intermediary. 

This technology has various properties, in particular the transparency of transactions, anonymity, the absence of a central authority (decentralization), and the limitation of the quantity in circulation. If someone could decipher the code to predict what Bitcoin's future prices would be, they would practically rule the world. 

As an AI and Finance enthusiast, this paper combines two of my areas of interest. In this article we will use ARIMA and LSTM to predict the price of BITCOIN.
##Agenda##

1- Load the Data
2- Exploratory Data Analysis
3- Autoregressive Integrated Moving Average ( ARIMA)
4- Long Short Term Memory
5- Conclusion
# # 1- Load the Data
# 
# There are many resources that we can use to obtain historical data. Our dataset comes from Yahoo Finance and covers all available data from 2014-09-17 to 2019-10-31 on the price of Bitcoin-USD.

# In[50]:


#import packages
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

#to plot within notebook
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 30,15 #width 20, height 10

#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))


# In[51]:


import os 
os.getcwd() # path


# In[52]:


#read the file
df = pd.read_csv('/Users/messidzochokki/Desktop/BTC-USD.csv', sep=',').fillna(0)

#print the head
df.head()


# # 2- Exploratory Data Analysis

# In[53]:


df.describe()


# In[54]:


df.info()

The data set contains several variables: date, Open, High, High, Low, Close ,Adj Close and Volume. 

The calculation of income or loss is generally determined by the closing price of a stock for the day, so we will consider the closing price as the target variable. 

Let's plot the target variable to understand how it is structured. 
# In[55]:


#setting index as date
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

ax = df.plot(x='Date', y='Close', style='b-', grid=True)
ax.set_xlabel("Date")
ax.set_ylabel("BITCOIN-USD")

The value of Bitcoin peaked in the last quarter of 2017 at nearly $20,000 and then declined sharply in early 2018. 

There was also a price increase in the third quarter of 2019. Now we are going to create an ARIMA model and will train it with the closing price of the Bitcoin on the train data. So let us split the data into train and test set and visualize it. 
# In[56]:


pip install pmdarima


# In[57]:


#from pyramid.arima import auto_arima
from pmdarima.arima import auto_arima
data = df.sort_index(ascending=True, axis=0)
data.head()

#split data
train, test = data[0:int(len(data)*0.8)], data[int(len(data)*0.8):]

#Plot data
ax = train.plot(x='Date', y='Close', style='b-', grid=True)
ax = test.plot(x='Date', y='Close', style='y-', grid=True, ax=ax)

ax.legend(['train','test'])
ax.set_xlabel("Date")
ax.set_ylabel("BITCOIN-USD")


# In[58]:


# show the shape of data
print(train.shape)
print(test.shape)
training = train['Close']
validation = test['Close']
print(validation.shape)


# # 3-Autoregressive Integrated Moving Average ( ARIMA)
ARIMA techniques are used to analyze time series and, in the past, have been mainly used for load forecasting, due to their accuracy and mathematical soundness. 

ARIMA est une technique très populaire pour la modélisation de séries chronologiques. Il décrit la corrélation entre les points de données et prend en compte la différence des valeurs.
Now we are going to use Auto ARIMA to get the best parameters, p, d and q

The parameters of the ARIMA model are defined as follows:
p: The number of lag observations included in the model, also called the lag order.
d: The number of times that the raw observations are differenced, also called the degree of differencing.
q: The size of the moving average window, also called the order of moving average.

Auto ARIMA: Automatically discover the optimal order for an ARIMA model
# In[59]:


model_autoARIMA = auto_arima(training, start_p=0, start_q=0,
                      test='adf',       # use adftest to find  optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
print(model_autoARIMA.summary())

The Auto ARIMA model provided the value of p,d, and q as 1,1 and 1 respectively

Before moving forward, let’s review the residual plots from auto ARIMA.
# In[60]:


model_autoARIMA.plot_diagnostics(figsize=(15,8))
plt.show()

So how to interpret the plot diagnostics?

**Top left**: The residual errors seem to fluctuate around a mean of zero and have a uniform variance. 

**Top Right**: The density plot suggest normal distribution with mean zero. 

**Bottom left**: All the dots should fall perfectly in line with the red line. Any significant deviations would imply the distribution is skewed. 

**Bottom Right**: The Correlogram, aka, ACF plot shows the residual errors are not autocorrelated.

Overall, it seems to be a good fit. Let’s start forecasting the stock prices. Next, create an ARIMA model with provided optimal parameters p, d and q.
# In[61]:


from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(training, order=(1, 1, 1))  
fitted = model.fit(disp=-1)  
print(fitted.summary())

Now let's start forecast the stock prices on the test dataset keeping 95% confidence level
# In[62]:


# Forecast
fc, se, conf = fitted.forecast(375, alpha=0.05)  # 95% confidence
fc_series = pd.Series(fc, index=validation.index)

lower_series = pd.Series(conf[:, 0], index=validation.index)
upper_series = pd.Series(conf[:, 1], index=validation.index)

plt.figure(figsize=(12,5), dpi=100)

plt.plot(training, color = 'blue', label='training')
plt.plot(validation, color = 'orange', label='Actual BITCOIN-USD Price')
plt.plot(fc_series, color = 'red',label='Predicted BITCOIN-USD Price')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.10)

plt.title('BITCOIN Price Prediction')
plt.xlabel('Date')
plt.ylabel('Actual BITCOIN-USD Price')

plt.legend(loc='upper left', fontsize=8)


    An auto ARIMA model uses past data to understand the pattern in the time series. Using these values, the model captured an increasing trend in the series, these predictions are still not close to the real values.Dans la suite de ce travail, nous utiliserons LSTM pour prévoir le cours de clôture quotidien du Bitcoin-USD
# # 4-Long Short Term Memory Network  (LSTM)
Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. They were introduced by Hochreiter & Schmidhuber (1997), and were refined and popularized by many people in following work. They work tremendously well on a large variety of problems, and are now widely used.

LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!Before to start , we need to install the necessary libraries in order to create our Neural Network (you may need to install Tensorflow Keras)
# In[63]:


import tensorflow as tf
from tensorflow.keras import layers


# In[64]:


#creating dataframe

new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]
#print(new_data.head())


# In[65]:


#setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

LSTMs expect our data to be in a specific format, usually a 3D array. We start by creating data in 30 timesteps (teps)and converting it into an array using NumPy. Next, we are split our data in train and test(valid)
# In[66]:


#creating train and test sets
dataset = new_data.values
teps =10

train, valid = dataset[0:int(len(dataset)*0.8)], dataset[int(len(dataset)*0.8)-teps:]

We’ll use Scikit- Learn’s MinMaxScaler and scale our dataset to numbers between zero and one
# In[67]:


#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)


# In[68]:


X_train, y_train = [], []
for i in range(teps,len(train)):
    X_train.append(scaled_data[i-teps:i,0])
    y_train.append(scaled_data[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))


Building the LSTM

Below is the code we use to train the model and do predictions.

In order to build the LSTM, we need to import a couple of modules from Keras: 

**Sequential for initializing the neural network LSTM 
**for adding the Long Short-Term Memory layer 
**Dense for adding a densely connected neural network layer
# In[69]:


from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense  
from tensorflow.keras.layers import LSTM  
#from tensorflow.keras.layers import Dropout 

from tensorflow.keras import backend


lstm_model = tf.keras.Sequential()
lstm_model.add(tf.keras.layers.LSTM(20, input_shape=(X_train.shape[1], 1),return_sequences=True,dropout=0.2))
lstm_model.add(tf.keras.layers.LSTM(20, return_sequences=True, dropout =0.2))
lstm_model.add(tf.keras.layers.LSTM(20))
lstm_model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))


lstm_model.compile(optimizer="adam", loss="mean_squared_error")

lstm_model.fit(X_train, y_train, epochs=50, batch_size = 4, verbose=2)


# In[21]:


lstm_model.summary()


# In[70]:


#predicting 246 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - teps:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(teps,inputs.shape[0]):
    X_test.append(inputs[i-teps:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = lstm_model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)


# In[71]:


rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
rms


# In[72]:


#for plotting
#train = new_data[:987]
#valid = new_data[987:]
train, valid = new_data[0:int(len(new_data)*0.8)], new_data[int(len(new_data)*0.8)-teps:]

valid['Predictions'] = closing_price




#plt.plot(train['Close'])
#plt.plot(valid[['Close','Predictions']])
ax = train['Close'].plot(x='Date', y='Close', style='b-', grid=True)
ax = valid['Close'].plot(x='Date', y='Close', style='y-', grid=True, ax=ax)
ax = valid['Predictions'].plot(x='Date', y='Predictions', style='r-', grid=True, ax=ax)
ax.legend(['train','test', 'Prediction'])
ax.set_xlabel("Date")
ax.set_ylabel("BITCOIN-USD")
#plt.savefig("BTC_predictions.png")


# In[73]:


# Visualising the results
plt.plot( valid['Close'], color = 'green', label = 'Real BITCOIN-USD Price')
plt.plot( valid['Predictions'], color = 'red', label = 'Predicted BITCOIN-USD Price')
plt.title('BITCOIN-USD Prediction')
plt.xlabel('Date')
plt.ylabel('BITCOIN-USD Price')
plt.legend()
plt.show()
plt.savefig("BTC_USD_predictions.png")

From the plot we can see that the real BITCOIN-USD price went up while our model also predicted that the price of the stock will go down. This clearly shows how powerful LSTMs are for analyzing time series and sequential data.
# # 5-Conclusion
Although the LSTM model above is not good enough to be used in live trading, the foundations built by developing such a model can help us build better models that might one day be used in our trading systemIf you have any questions leave them in the comment section and I we can discuss them. Thank you!😊
# **References**
# https://colah.github.io/posts/2015-08-Understanding-LSTMs/
# 
# https://machinelearningmastery.com/save-arima-time-series-forecasting-model-python/
# 
# 

# https://acadpubl.eu/hub/2018-119-17/3/214.pdf

# In[ ]:




