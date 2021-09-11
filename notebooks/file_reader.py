import pandas as pd


def read_file(path='../Lecture/1.Linear_Regression/FuelConsumptionCo2.csv'):
    return pd.read_csv(path, nrows=5)


if __name__ == '__main__':
    print(read_file())