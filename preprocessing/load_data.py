import pandas as pd
from pathlib import Path


class DataLoader:
    def __init__(self, data_folder):
        self.data_folder = Path(data_folder)

    def load_turbine(self, turbine_name):
        file_path = self.data_folder / f"{turbine_name}.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found.")

        df = pd.read_csv(file_path, sep=";")

        return df