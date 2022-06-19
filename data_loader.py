# %% Load the data from downloaded CSV files
import pandas as pd


def load_data():
    train = pd.read_csv(r"CSV\train.csv", index_col="PassengerId")
    test = pd.read_csv(r"CSV\test.csv", index_col="PassengerId")
    return train, test


def split_data(data, to_predict_column):
    X = data.copy()
    y = X.pop(to_predict_column)
    return X, y
