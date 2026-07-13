from preprocessing.load_data import DataLoader
from preprocessing.data_cleaner import DataCleaner

loader = DataLoader('data/raw/datasets')
cleaner = DataCleaner()

for name in loader.list_turbines():
    df = cleaner.clean(loader.load_turbine(name))
    dates = df['time_stamp'].dt.date.astype(str)
    print(name, dates.min(), dates.max(), len(dates.unique()))
