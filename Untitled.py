#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import pandas as pd
import numpy as np
import math
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from itertools import cycle
import plotly.express as px


# In[2]:


def stock():
    global stock_name
    stock_name = input("\nEnter a valid stock name (e.g. \"TCS.NS\" etc)\nGo to \"https://finance.yahoo.com/quote/MARUTI.NS/history\" to get the HISTORICAL DATA:\t")
    print(f"The stock chart of {stock_name} is downloading......\n")
    global data
    data = yf.download(stock_name, period='max')
    print(data)

stock()

while True:
    repeat = input('\n\nIs it correct exactly the same what do you want?\n if YES, then type \"yes\"\nOR\n if NO, then type \"no\" \n\t: ')
    if repeat.lower() == 'no':
        stock()
    else:
        print('\n..................... thanks .....................')
        break

data['Date'] = data.index
data = data[['Date', 'Open', 'High', 'Low', 'Close']]  # choose only relevant features for the model
data.reset_index(drop=True, inplace=True)


# In[3]:


# Visualize close price
closedf = data[['Date','Close']]
fig = px.line(closedf, x=closedf.Date, y=closedf.Close, labels={'Date':'Date','Close':'Close Stock'})
fig.update_traces(marker_line_width=2, opacity=0.8, marker_line_color='orange')
fig.update_layout(title_text=f'{stock_name} Stock Chart', plot_bgcolor='white', font_size=15, font_color='black')
fig.show()


# In[4]:


StartDate = data.iloc[0][0]
EndDate = data.iloc[-1][0]
data.dropna(inplace=True)

data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
closedf = data[['Date', 'Close']]
closedf = closedf[closedf['Date'] > '2022-01-01']
close_Stock = closedf.copy()


# In[5]:


# Cell 3: Prepare Data for LSTM Model
scaler = MinMaxScaler(feature_range=(0, 1))
closedf_scaled = scaler.fit_transform(np.array(closedf['Close']).reshape(-1, 1))

training_size = int(len(closedf_scaled) * 0.60)
train_data, test_data = closedf_scaled[0:training_size], closedf_scaled[training_size:len(closedf_scaled)]

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        dataX.append(dataset[i:(i + time_step), 0])
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 15
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# In[6]:


# Cell 4: Build and Train the LSTM Model
model = Sequential()
model.add(LSTM(10, input_shape=(time_step, 1), activation="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=32, verbose=1)


# In[7]:


train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate and print evaluation metrics
print("Train data RMSE: ", math.sqrt(mean_squared_error(original_ytrain, train_predict)))
print("Train data MSE: ", mean_squared_error(original_ytrain, train_predict))
print("Train data MAE: ", mean_absolute_error(original_ytrain, train_predict))
print("Test data RMSE: ", math.sqrt(mean_squared_error(original_ytest, test_predict)))
print("Test data MSE: ", mean_squared_error(original_ytest, test_predict))
print("Test data MAE: ", mean_absolute_error(original_ytest, test_predict))
print("Train data explained variance score:", explained_variance_score(original_ytrain, train_predict))
print("Test data explained variance score:", explained_variance_score(original_ytest, test_predict))
print("Train data R2 score:", r2_score(original_ytrain, train_predict))
print("Testdata R2 score:", r2_score(original_ytest, test_predict))

# Calculate prediction accuracy percentage
accuracy_percentage = (1 - (mean_absolute_error(original_ytest, test_predict) / np.mean(original_ytest))) * 100
print("Prediction Accuracy Percentage:", accuracy_percentage)


# In[8]:


look_back = time_step
trainPredictPlot = np.empty_like(closedf_scaled)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

testPredictPlot = np.empty_like(closedf_scaled)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(closedf_scaled) - 1, :] = test_predict

names = cycle(['Original close price', 'Train predicted close price', 'Test predicted close price'])

plotdf = pd.DataFrame({'date': close_Stock['Date'],
                       'original_close': close_Stock['Close'],
                       'train_predicted_close': trainPredictPlot.reshape(-1),
                       'test_predicted_close': testPredictPlot.reshape(-1)})

fig = px.line(plotdf, x=plotdf['date'], y=['original_close', 'train_predicted_close', 'test_predicted_close'],
              labels={'value': 'Stock price', 'date': 'Date'})
fig.update_layout(title_text='Comparison between original close price vs predicted close price',
                  plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
fig.for_each_trace(lambda t: t.update(name=next(names)))
fig.show()


# In[ ]:


x_input = test_data[len(test_data) - time_step:].reshape(1, -1)
temp_input = list(x_input)
temp_input = temp_input[0].tolist()

lst_output = []
i = 0
pred_days = 30
while i < pred_days:
    if len(temp_input) > time_step:
        x_input = np.array(temp_input[1:])
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, time_step, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
        i += 1
    else:
        x_input = x_input.reshape((1, time_step, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i += 1

last_days = np.arange(1, time_step + 1)
day_pred = np.arange(time_step + 1, time_step + pred_days + 1)
temp_mat = np.empty((len(last_days) + pred_days + 1, 1))
temp_mat[:] = np.nan
temp_mat = temp_mat.reshape(1, -1).tolist()[0]

last_original_days_value = temp_mat
next_predicted_days_value = temp_mat

last_original_days_value[0:time_step] = scaler.inverse_transform(closedf_scaled[len(closedf_scaled) - time_step:]).reshape(1, -1).tolist()[0]
next_predicted_days_value[time_step:] = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)).reshape(1, -1).tolist()[0]

new_pred_plot = pd.DataFrame({
    'last_original_days_value': last_original_days_value,
    'next_predicted_days_value': next_predicted_days_value
})

names = cycle(['Last 15 days close price', 'Predicted next 30 days close price'])

fig = px.line(new_pred_plot, x=new_pred_plot.index, y=['last_original_days_value', 'next_predicted_days_value'],
              labels={'value': 'Stock price', 'index': 'Timestamp'})
fig.update_layout(title_text='Compare last 15 days vs next 30 days',
                  plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
fig.for_each_trace(lambda t: t.update(name=next(names)))
fig.show()

