import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hw1_modules import *


# read data from CSV to array
train_values = np.array(pd.read_csv("train.csv").values)
test_data = np.array(pd.read_csv("test.csv").values)

print(test_data.shape)
test_data1 = test_data[0:5000,:]
test_data2 = test_data[5000:10000,:]
test_data3 = test_data[10000:15000,:]
test_data4 = test_data[15000:20000,:]
test_data5 = test_data[20000:25000,:]
test_data6 = test_data[25000:,:]
del test_data

#separate values and labels into separate arrays, don't duplicate full dataset
train_labels = train_values[:,0]
train_values = train_values[:,1:]

#convert labels array to vertical, 2d array of one column
train_labels = np.expand_dims(train_labels, axis=1)


print(train_values.shape)
print(train_labels.shape)
print(test_data1.shape)

number_neighbors = 3

predictions1 = knn_predict_class(train_values, train_labels, test_data1, number_neighbors)
predictions2 = knn_predict_class(train_values, train_labels, test_data2, number_neighbors)
predictions3 = knn_predict_class(train_values, train_labels, test_data3, number_neighbors)
predictions4 = knn_predict_class(train_values, train_labels, test_data4, number_neighbors)
predictions5 = knn_predict_class(train_values, train_labels, test_data5, number_neighbors)
predictions6 = knn_predict_class(train_values, train_labels, test_data6, number_neighbors)
#predictions = pd.concat(predictions, pd.DataFrame(predictions1))
print(predictions1.shape)
print(predictions1)

predictions = np.concatenate((predictions1, predictions2, predictions3, predictions4, predictions5, predictions6))
print(predictions.shape)

predictions_df = pd.DataFrame(predictions, columns=["Labels"])

print(predictions_df)
predictions_df.to_csv("test_predictions.csv", header=True, index_label="ImageId")