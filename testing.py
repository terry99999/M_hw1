import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

digit_source_array = np.array([1,0,16,7,3,8,21,6,10,11])

digit_test = pd.DataFrame(np.concatenate([np.array([0,1,2,3,4,5,6,7,8,9]).reshape((10,1)), digit_source_array.reshape((10,1)), np.full([10,1], np.NaN),np.full([10,1], np.inf),np.full([10,1], np.nan)], axis=1), columns=["Digit", "SourceIdx", "MatchIdx", "MatchDist", "MatchLabel"])
digit_test["Digit"] = digit_test["Digit"].astype(int)

print(digit_test)
print(digit_test.iloc[3,1])

#making changes
#different