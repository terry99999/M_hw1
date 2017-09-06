import pandas as pd
import numpy as np
import scipy as sp
import scipy.spatial.distance as spd

data = pd.read_csv("train.csv", dtype={"label" : "category"})


# index of digits 0-9 in data manually selected [1,0,16,7,3,8,21,6,10,11]

digits = data.iloc[[1,0,16,7,3,8,21,6,10,11],:]

results = spd.cdist(digits, data, "euclidean")

results[results==0] = np.inf

mins_location = np.argmin(results, axis=1)

digit_test = pd.DataFrame(np.concatenate([np.array([0,1,2,3,4,5,6,7,8,9]).reshape((10,1)), np.array([1,0,16,7,3,8,21,6,10,11]).reshape((10,1)), np.full([10,1], np.NaN),np.full([10,1], np.nan)], axis=1), columns=["Digit", "SourceIdx", "MatchIdx", "MatchLabel"], dtype=int)

for i in range (0, 10):
    print(mins_location[i])
    digit_test.iloc[i,2] = mins_location[i]
    digit_test.iloc[i,3] = data.iloc[mins_location[i],0]

print(digit_test)
