import pandas as pd
import numpy as np 
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

model = load_model("forecaster.h5")

st.header("Stock Market Predictor")

stock = st.text_input("Enter Stock Symbol","AAPL")
start = "2011-1-1"
end = "2024-9-1"
data = yf.download(stock,start,end)

st.subheader("Stock Data")
st.write(data)
data.reset_index(inplace=True)
data_train = data.Close.loc[:int(data.shape[0]*0.8)-1]
data_test = data.Close.loc[int(data_train.shape[0] * 0.8):]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
past_100_days = data_train.tail(100)
data_test = pd.concat((past_100_days,data_test),ignore_index=True)

data_test_scale = scaler.fit_transform(pd.DataFrame(data_test))

st.subheader("Price vs MA50")
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(10,8))
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(ma_50_days,'r')
ax1.plot(data.Close,'g')
st.pyplot(fig1)


st.subheader("Price vs MA50 vs MA100")
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(10,8))
ax2 = fig2.add_subplot(1, 1, 1)
ax2.plot(ma_100_days,'b')
ax2.plot(ma_50_days,'r')
ax2.plot(data.Close,'g')
st.pyplot(fig2)

st.subheader("Price vs MA50 vs MA100 vs MA200")
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(10,8))
ax3 = fig3.add_subplot(1, 1, 1)
ax3.plot(ma_50_days,'r')
ax3.plot(data.Close,'g')
ax3.plot(ma_100_days,'b')
ax3.plot(ma_200_days,'k')
st.pyplot(fig3)


x = []
y = []
for i in range(100,data_test.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x),np.array(y)

predict = model.predict(x)
scale = 1 / scaler.scale_
predict = predict * scale

y = y * scale

st.subheader("Original Price vs Predicted Price")
fig4 = plt.figure(figsize=(10,8))
ax4 = fig1.add_subplot(1, 1, 1)
ax4.set_xlabel("Time")
ax4.set_ylabel("Price")
ax4.plot(ma_50_days,'r')
ax4.plot(data.Close,'g')
st.pyplot(fig4)