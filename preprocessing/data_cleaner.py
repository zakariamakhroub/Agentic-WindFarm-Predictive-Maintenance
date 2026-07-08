import pandas as pd


class DataCleaner:

    def clean(self, df):
        """
        Clean the SCADA dataset.
        """

        # 1. Convert timestamp to datetime
        df["time_stamp"] = pd.to_datetime(df["time_stamp"])

        # 2. Sort by time
        df = df.sort_values("time_stamp")

        # 3. Fill missing values
        df = df.ffill().bfill()

        # 4. Remove identifier columns
        df = df.drop(columns=["asset_id", "id"])

        # Reset index
        df = df.reset_index(drop=True)

        return df