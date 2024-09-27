import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import iplot,plot
import plotly.express as px
import scipy
import yfinance as yf
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm
from warnings import filterwarnings


ticker = 'AAPL'
start = "2011-1-1"
end = "2024-9-1"
data = yf.download(ticker,start=start, end=end)
data["Month"] = data.index.month_name().values
month_order = ["January", "February", "March", "April", "May", 
               "June", "July", "August", "September", "October", 
               "November", "December"]
data["Month"] = pd.Categorical(data["Month"],categories=month_order,ordered=True)
data["year"] = data.index.year
pivot1 = pd.pivot_table(data=data,values="Close",index="year",columns="Month",aggfunc=["mean"],observed=True)
pivot2 = pd.pivot_table(data=data,values="Close",index="year",columns="Month",aggfunc=["median"],observed=True)
pivot3 = pd.pivot_table(data=data,values="Close",index="year",columns="Month",aggfunc=["max","min"],observed=True)
f,ax = plt.subplots(data.shape[1]-2,1,figsize=(12,35))
for i,c in enumerate(data.columns[:-2]):
    ax[i].plot(data.index, data[c])
    ax[i].set_title(f"{c} Price Over the years",weight='bold',fontsize=16)
    ax[i].set_xlabel("Year")
    ax[i].set_ylabel("Price")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,6))
plt.title("Open vs Adj Close",weight='bold',fontsize=16)
plt.plot(data.index,data.Open,color='blue')
plt.plot(data.index,data["Adj Close"],color='orange')
plt.xlabel("Year")
plt.ylabel("Price")
plt.show()

f,ax = plt.subplots(figsize=(8,6))
sns.boxplot(data=data[["Open","High","Low","Close","Adj Close"]],palette='autumn',ax=ax)
plt.show()

filterwarnings("ignore")

ax = plt.subplots(figsize=(12,6))
sns.swarmplot(data=data,x='Month',y='Open',hue='year',palette='flare',ax=ax[1])
ax[1].legend(shadow=True,bbox_to_anchor=(1.02,1))
plt.show()

ma_50 = data.Close.rolling(50).mean()
plt.figure(figsize=(14,8))
plt.title("MA50")
plt.plot(ma_50,color='r')
plt.plot(data.Close,color='b')
plt.show()

ma_100 = data.Close.rolling(100).mean()
plt.figure(figsize=(14,8))
plt.title("MA50 vs MA100")
plt.plot(ma_50,color='r')
plt.plot(data.Close,color='g')
plt.plot(ma_100,color='b')
plt.show()

df = data["Close"]
f,ax = plt.subplots(figsize=(12,8))
plot_acf(df,lags=300,ax=ax)
plt.ylim(-1,1.1)
plt.show()

f,ax = plt.subplots(figsize=(12,8))
plot_pacf(df,lags=15,ax=ax)
plt.ylim(-1,1.1)
plt.show()

# Checking for Stationarity
adf_res = adfuller(df)

print(f'ADF Statistic: {adf_res[0]}')
print(f'p-value: {adf_res[1]}')
print(f'Critical Values:')
for key, value in adf_res[4].items():
    print(f'   {key}: {value}')

df2 = df.asfreq('D')
df2 = df2.fillna(method='ffill')
plt.figure(figsize=(12,8))
df2.plot()
from statsmodels.tsa.seasonal import seasonal_decompose
dec = seasonal_decompose(df2)
trend = dec.trend
residuals = dec.resid
seasonal = dec.seasonal

f,ax = plt.subplots(4,1,figsize=(10,18))
ax[0].plot(df2)
ax[0].set_title("Time Series Decomposition")
ax[0].set_ylabel("Original")
ax[1].plot(trend)
ax[1].set_ylabel("Trend")
ax[2].plot(residuals)
ax[2].set_ylabel("Residuals")
ax[3].plot(seasonal)
ax[3].set_ylabel("Seasonal")
plt.tight_layout()
plt.show()

df.diff().dropna().plot()
plt.show()

# pip install pmdarima

from pmdarima.arima.utils import ndiffs
print(ndiffs(df2,test='adf'))
df_diff = df2.diff().dropna()
result = adfuller(df_diff)
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
print(f'Critical Values:')
for key, value in result[4].items():
    print(f'   {key}: {value}')

plot_acf(df_diff)
plt.ylim(-1,1.05)
plt.show()

plot_pacf(df_diff)
plt.ylim(-1,1.05)
plt.show()

train = df_diff.iloc[:int(df_diff.shape[0]*0.8)]
test = df_diff.iloc[int(df_diff.shape[0] * 0.8):]

plt.plot(train)
plt.plot(test)
plt.show()

import pmdarima as pm
model = pm.auto_arima(train,seasonal=True)
print(model.summary())

from statsmodels.tsa.statespace.sarimax import SARIMAX
model_sar = SARIMAX(train,order=(5,1,0))
model_fit = model_sar.fit()
preds = model_fit.predict(start=test.index[0],end=test.index[-1])

plt.figure(figsize=(10,6))
plt.plot(preds,color='b',alpha=0.5)
plt.plot(test,color='k',alpha=0.5)
plt.show()

train = df_diff.iloc[:int(df_diff.shape[0]*0.8)]
tr = train
preds = pd.Series()
for i in range(0,10):
    model2 = SARIMAX(tr,order=(5,1,0))
    model_fit2 = model2.fit()
    pred = model_fit.predict(test.index[i])
    preds[test.index[i]] = pred
    tr = pd.concat((tr,pred))

plt.figure(figsize=(12,6))
plt.plot(preds,color='b',alpha=0.5)
plt.plot(test.iloc[:10],color='k',alpha=0.5)
plt.show()

from scipy.stats import boxcox
log_data2,lambda_ = boxcox(df)

plt.figure(figsize=(8,6))
plt.plot(log_data2,color='blue')
plt.show()

print(lambda_)
df4 = pd.Series(data=log_data2,index=df.index).diff().dropna()

df4.plot()
plt.show()

adf_res2 = adfuller(df4)
print(f'ADF Statistic: {adf_res2[0]}')
print(f'p-value: {adf_res2[1]}')
print(f'Critical Values:')
for key, value in adf_res2[4].items():
    print(f'   {key}: {value}')

if adf_res2[1] < 0.05:
    print("Our data is Stationary")
else:
    print("data is non Stationary")

f,ax = plt.subplots(figsize=(8,6))
plot_acf(df4,lags=15,ax=ax)
plt.ylim(-1,1.1)
plt.show()

f,ax = plt.subplots(figsize=(8,6))
plot_pacf(df4,lags=15,ax=ax)
plt.ylim(-1,1.1)
plt.show()

train_end = datetime(2024,1,4)
test_end = datetime(2024,9,1)
train_data = df4[:train_end - timedelta(days=1)]
test_data = df4[train_end:]
model = ARIMA(train_data.values,order=(9,0,9))
model_fit = model.fit()
print(model_fit.summary())
pred_start_date = test_data.index[0] - timedelta(days=1)
pred_end_date = test_data.index[-1]
preds = model_fit.predict(start=3273,end=3273+test_data.shape[0]-1)
s_pred = pd.Series(data=preds,index=test_data.index)

plt.figure(figsize=(10,6))
plt.plot(s_pred)
plt.plot(test_data)
plt.show()

data.reset_index(inplace=True)
ma_100 = data.Close.rolling(100).mean()
plt.figure(figsize=(12,6))
plt.plot(data.Close)
plt.plot(ma_100,color='r')
plt.show()

ma_200 = data.Close.rolling(200).mean()
plt.figure(figsize=(12,6))
plt.plot(data.Close)
plt.plot(ma_200,color='r')
plt.plot(ma_100,color='g')
plt.show()

train_data = data.Close.loc[0: int(data.shape[0] * 0.8)-1]
test_data = data.Close.loc[int(data.shape[0] * 0.8):]
from sklearn.preprocessing import MinMaxScaler
print(train_data.values)
mx = MinMaxScaler()
train_data_scaled = mx.fit_transform(train_data.values.reshape(-1,1))
print(train_data_scaled[0:50])
x = []
y = []
for i in range(100,train_data_scaled.shape[0]):
    x.append(train_data_scaled[i-100:i])
    y.append(train_data_scaled[i,0])
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dropout,Dense
x,y = np.array(x),np.array(y)
model = Sequential()
model.add(LSTM(units=50,activation='relu',return_sequences=True,input_shape=(x.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=60,activation='relu',return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=80,activation='relu',return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=120,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1))
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x,y,epochs=50,batch_size=32,verbose=1)
print(model.summary())
past_100_days = train_data.tail(100)
test_data = pd.concat((past_100_days,test_data),ignore_index=True)
test_data_scaled = mx.fit_transform(pd.DataFrame(test_data))
x = []
y = []
for i in range(100,test_data_scaled.shape[0]):
    x.append(test_data_scaled[i-100:i])
    y.append(test_data_scaled[i,0])

x,y = np.array(x),np.array(y)
y_pred = model.predict(x)
scale = 1 / mx.scale_
y_pred = y_pred * scale
y = y * scale

plt.figure(figsize=(10,8))
plt.plot(y_pred,color='r',label="Predicted Price")
plt.plot(y,color='g',label="Original Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()

model.save("forecaster.h5")