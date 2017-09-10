import numpy as np
import pandas as pd
import scipy.spatial.distance as spd
import scipy.stats as sps
import sklearn.model_selection as skm
import sklearn.metrics as skmetrics
import matplotlib.pyplot as plt
import seaborn as sb
from hw1_modules import *

# read data from CSV to array
data = np.array(pd.read_csv("train.csv").values)

#separate values and labels into separate arrays
values = data[:,1:]
labels = data[:,0]
#convert labels array to vertical, 2d array of one column
labels = np.expand_dims(labels, axis=1)

#initialize confusion matrix
cf = np.ones((10,10), dtype=int)
#initialize cumulative accuracy
accuracy = 0
#set number of folds
number_folds = 3
#set k for k neighbors
k_neighbors = 3

#create kfold iterating object
kf = skm.KFold(n_splits=number_folds)
for train_idx, test_idx in kf.split(values, labels):
    print("Dividing data")
    #subset data using indexes generated by kfold object
    train_data = values[train_idx]
    test_data = values[test_idx]
    train_labels = labels[train_idx]
    test_lables = labels[test_idx]
    #run one iterating of testing with knn
    print("Testing data")
    predicted_labels = knn_predict_class(train_data, train_labels, test_data, k_neighbors)
    print("Accuracy for this run" + str(sum(predicted_labels == test_lables)/len(test_lables)))
    #cumulative accuracy
    accuracy += sum(predicted_labels == test_lables)/len(test_lables)
    #add this run's confusion values to cumulative confusion matrix
    cf = cf + skmetrics.confusion_matrix(test_lables, predicted_labels)

#calculate average accuracy from cumulative
accuracy = accuracy/number_folds
print(accuracy)

#Create and display plot for confusion matrix
ax = sb.heatmap(cf, annot=True, fmt="d")
ax.set(xlabel="Predicted Label", ylabel="True Label")
plt.show()