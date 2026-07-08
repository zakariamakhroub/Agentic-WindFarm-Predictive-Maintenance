import pandas as pd


class DataExplorer:

    @staticmethod
    def explore(df):

        print("=" * 50)
        print("DATASET SHAPE")
        print("=" * 50)
        print(df.shape)

        print("\n")

        print("=" * 50)
        print("COLUMN NAMES")
        print("=" * 50)
        print(df.columns.tolist())

        print("\n")

        print("=" * 50)
        print("DATA TYPES")
        print("=" * 50)
        print(df.dtypes)

        print("\n")

        print("=" * 50)
        print("MISSING VALUES")
        print("=" * 50)
        print(df.isnull().sum().sort_values(ascending=False).head(20))

        print("\n")

        print("=" * 50)
        print("DUPLICATED ROWS")
        print("=" * 50)
        print(df.duplicated().sum())