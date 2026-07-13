import os
import pandas as pd


class DataLoader:

    def __init__(self, data_path):
        self.data_path = data_path

    def _normalize_turbine_name(self, turbine_name):
        normalized = turbine_name.strip().lower().replace(" ", "")
        if normalized.startswith("turbine") and normalized[7:].isdigit():
            return f"turbine{int(normalized[7:])}"
        return normalized

    def load_turbine(self, turbine_name):
        normalized_name = self._normalize_turbine_name(turbine_name)
        file_path = os.path.join(self.data_path, f"{normalized_name}.csv")
        return pd.read_csv(file_path, sep=";")

    def list_turbines(self):
        return [
            self._normalize_turbine_name(file.replace(".csv", ""))
            for file in os.listdir(self.data_path)
            if file.endswith(".csv")
        ]