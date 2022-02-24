from pickle import TRUE
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


data = pd.read_csv("FinalData.csv", sep = ";")
#print(data.iloc[1:10])
training_set = data.iloc[1:3000, 2:3]



sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)




X_train = []
y_train = []
for i in range(60, 2035):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

print(X_train)

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


#p1 = data.loc[(data["particpant_ID"] == 1) & (data["Round"] == 1) & (data["Phase"] == 2)]
#date_time_obj = datetime.strptime(data["time"][1], '%y-%m-%d %H:%M:%S')

#print(date_time_obj)

targetData = pd.read_csv("TargetData.csv", sep =";")
frus = targetData["frustrated"]



plt.hist(targetData["frustrated"], bins=7, rwidth= 0.8)


