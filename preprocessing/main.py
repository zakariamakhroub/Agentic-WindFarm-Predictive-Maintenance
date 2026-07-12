from preprocessing.data_cleaner import DataCleaner
from preprocessing.data_explorer import DataExplorer
from preprocessing.load_data import DataLoader
from preprocessing.data_splitter import DataSplitter
from preprocessing.target_creator import TargetCreator

import pandas as pd
loader = DataLoader("data/raw/datasets")

df = loader.load_turbine("turbine 0")


cleaner = DataCleaner()
splitter = DataSplitter()
target_creator = TargetCreator()


clean_df = cleaner.clean(df)



train_df, prediction_df = splitter.split(clean_df)

train_df = target_creator.create_target(train_df)
prediction_df = target_creator.create_target(prediction_df)

print(train_df["target"].value_counts())
print("Training:", train_df.shape)
print("Prediction:", prediction_df.shape)

