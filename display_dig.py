from hw1_modules import *

data = pd.read_csv("train.csv", dtype={"label" : "category"})

for i in (range(0, 10)):
    for j in range(0,len(data)-1):
        if int(data.iloc[j,:].loc["label"]) == i:
            displayDigit(data.iloc[j, 1:])
            break