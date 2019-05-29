import numpy as np 
from sklearn.linear_model import LinearRegression
import pandas as pd



def main():
    train_data = np.zeros((5, 5))
    # train_data.I
    # print(train_data.I)

    test = pd.read_csv("test_set.csv")
    test.to_csv("cyd.csv")
    return 1
if __name__ == "__main__":
    main()