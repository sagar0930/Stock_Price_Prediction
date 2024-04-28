import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense

def data_scaling(final_df):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    y_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    dataset = min_max_scaler.fit_transform(final_df[['Open', 'High', 'Low','Volume','headline_sentiment', 'summary_sentiment']])
    y_scaled = y_scaler.fit_transform(final_df[['Close']])
    dataset = np.concatenate((dataset, y_scaled), axis=1)
    return final_df, dataset, y_scaler


def create_sequences(dataset, look_back=15):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 4]) # column with index 4 is close value which we want to predict
    return np.array(dataX), np.array(dataY)


def train_test_split(dataset, look_back=15):
    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    x_train, y_train = create_sequences(train, look_back)
    x_test, y_test = create_sequences(test, look_back)
    return x_train, y_train, x_test, y_test

def fit_and_evaluate_model(x_train, x_test, y_train, y_test,y_scaler):
    model = Sequential()
    model.add(LSTM(10, input_shape=(15, 7))) # hyperbolic tangent (tanh) activation function used by default
    model.add(Dense(1)) # linear activation function is used by default
    model.compile(loss='mean_squared_error', optimizer='adam')
    #'adam' is a popular optimization algorithm that adapts the learning rates of each parameter individually, 
    #making it well-suited for a wide range of tasks. Adam is an extension of stochastic gradient descent (SGD) 
    #and is known for its efficiency in training deep neural networks.
    model.summary()
    model.fit(x_train, y_train, epochs=20, batch_size=10, verbose=2)
    
    trainPredict = model.predict(x_train)
    testPredict = model.predict(x_test)
    trainPredict = y_scaler.inverse_transform(trainPredict)
    trainY = y_scaler.inverse_transform([y_train])
    testPredict = y_scaler.inverse_transform(testPredict)
    testY = y_scaler.inverse_transform([y_test])

    return model , trainY, trainPredict


def errors(prediction,actual): 
    mae = np.mean(np.abs(prediction - actual))
    mape = np.mean(np.abs(prediction - actual)/np.abs(actual))*100
    rmse = np.mean((prediction - actual)**2)**0.5
    return mae, mape, rmse



