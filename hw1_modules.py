import pandas as pd
import numpy as np

def displayDigit(row):
    digit_array = np.array(row)
    digit_array.resize((28,28))
    plt.imshow(digit_array.astype(np.float32), cmap="binary")
    plt.show()

def l2distance(item1, item2):
    '''each item is 785 element series, label first then data'''
    return np.sqrt(sum(np.square(abs(np.array(item1[1:]) - np.array(item2[1:])))))