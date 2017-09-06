import pandas as pd
import numpy as np
import scipy as sp
import scipy.spatial.distance as spd
import itertools as it
import hw1_modules


data = pd.read_csv("train.csv", dtype={"label" : "category"})


values = {"label" : ["0","1"]}
data_01 = data[data.isin(values).any(1)]
distances = spd.pdist(data_01, metric="euclidean")
print(len(distances))
tuples = (list(it.combinations(range(data_01.shape[0]),2)))

with open('binary_dist.txt', 'w') as file_out:
    for i in range(0, len(tuples)):
        if i % 10000 == 0:
            print(i)
        result = str((data_01.iat[tuples[i][0], 0] == data_01.iat[tuples[i][1], 0]))
        file_out.write(result + "," + str(distances[i])+ "\n")
