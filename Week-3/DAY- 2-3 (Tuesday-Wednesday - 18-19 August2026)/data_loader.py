import pandas as pd


def load_dataset(path):

    return pd.read_csv(path)


def show_basic_info(df):

    print("\nDataset Information")
    print("-------------------")

    print(df.head())

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns)

    print("\nData Types:")
    print(df.dtypes)