import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def displayDigit(row):
    digit_array = np.array(row)
    digit_array.resize((28,28))
    plt.imshow(digit_array.astype(np.float32), cmap="binary")
    plt.show()

def l2distance(item1, item2):
    '''each item is 785 element series, label first then data'''
    return np.sqrt(sum(np.square(abs(np.array(item1[1:]) - np.array(item2[1:])))))


small_data = pd.read_csv("train_small.csv", dtype={"label" : "category"})
med_data = pd.read_csv("train_med.csv", dtype={"label" : "category"})
#print(small_data.head())

data = pd.read_csv("train.csv", dtype={"label" : "category"})

#small_data["label"] = small_data["label"].astype("category")


'''
for i in (range(0, 10)):
    for j in range(0,1000):
        if int(small_data.iloc[j,:].loc["label"]) == i:
            displayDigit(small_data.iloc[j, 1:])
            break


plt.hist(data["label"].astype(int), rwidth=.5, normed=True, bins=10)
plt.xticks(np.arange(0,10, 1))
plt.show()
'''



digit_source_array = np.array([1,0,16,7,3,8,21,6,10,11])

digit_test = pd.DataFrame(np.concatenate([np.array([0,1,2,3,4,5,6,7,8,9]).reshape((10,1)), digit_source_array.reshape((10,1)), np.full([10,1], np.NaN),np.full([10,1], np.inf),np.full([10,1], np.nan)], axis=1), columns=["Digit", "SourceIdx", "MatchIdx", "MatchDist", "MatchLabel"])
digit_test["Digit"] = digit_test["Digit"].astype(int)




for i in range (0, digit_test.shape[0]):
    print("Evaluating " + str(i))
    for j in range (0, data.shape[0]):
#        print(str(i) + " , " + str (j))
        distance = l2distance(data.iloc[int(digit_test.iat[i,1])], data.iloc[j])
        if distance == 0:
            continue
        elif distance < digit_test.iat[i,3]:
            digit_test.iat[i,3] = distance
            digit_test.iat[i,2] = j
            digit_test.iat[i,4] = data.iat[j,0]
        else:
            continue


print(digit_test)