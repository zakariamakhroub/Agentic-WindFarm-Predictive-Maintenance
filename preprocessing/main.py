from preprocessing.data_cleaner import DataCleaner
from preprocessing.data_explorer import DataExplorer
from preprocessing.load_data import DataLoader
from preprocessing.data_splitter import DataSplitter
import pandas as pd
loader = DataLoader("data/raw/datasets")

df = loader.load_turbine("turbine 0")


cleaner = DataCleaner()
splitter = DataSplitter()


clean_df = cleaner.clean(df)

DataExplorer.explore(clean_df)

train_df, prediction_df = splitter.split(clean_df)

print("Training:", train_df.shape)
print("Prediction:", prediction_df.shape)
