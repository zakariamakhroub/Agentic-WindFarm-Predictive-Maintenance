from preprocessing.load_data import DataLoader

loader = DataLoader("data/raw/datasets")

df = loader.load_turbine("turbine 0")

print(df.head())

print()

print(df.info())