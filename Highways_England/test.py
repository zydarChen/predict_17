from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


df = pd.read_csv('data/HE_2013_M25LM326_6_10.csv', delimiter=';', parse_dates=['date'])

scaler = StandardScaler()
values = df['travel_time'].values
print(values)
data = scaler.fit_transform(values[:, np.newaxis]).flatten()
print(data[-10:])
# data = scaler.inverse_transform(data)
# print(data[-10:])

sequence_length = 1 + 1
result = []
for index in range(len(data) - sequence_length + 1):
    result.append(data[index: index + sequence_length])

result = np.array(result)
row_8 = round(0.8 * result.shape[0])
row_9 = round(0.9 * result.shape[0])
train = result[:int(row_8), :]
x_train = train[:, :-1]
y_train = train[:, -1]
x_test = result[int(row_9):, :-1]
y_test = result[int(row_9):, -1]
print(x_test[-10:].flatten())
print(scaler.inverse_transform(y_test[-10:]))
