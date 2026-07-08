from preprocessing.data_cleaner import DataCleaner
from preprocessing.data_explorer import DataExplorer
from preprocessing.load_data import DataLoader
import pandas as pd
loader = DataLoader("data/raw/datasets")

df = loader.load_turbine("turbine 0")


cleaner = DataCleaner()


clean_df = cleaner.clean(df)

DataExplorer.explore(clean_df)

print(clean_df.info())
print(df["status_type_id"].value_counts())

print(clean_df["train_test"].value_counts())