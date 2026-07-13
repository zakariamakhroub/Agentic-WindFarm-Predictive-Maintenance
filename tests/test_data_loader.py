import pandas as pd

from preprocessing.load_data import DataLoader


def test_list_turbines_normalizes_names(tmp_path):
    csv_path = tmp_path / "turbine 1.csv"
    csv_path.write_text("temperature,voltage\n10,20\n", encoding="utf-8")

    loader = DataLoader(str(tmp_path))

    assert loader.list_turbines() == ["turbine1"]

    df = loader.load_turbine("turbine1")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["temperature", "voltage"]
