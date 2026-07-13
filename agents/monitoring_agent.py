import pandas as pd


class MonitoringAgent:
    """
    Selects the SCADA observations for a chosen day.
    """

    def __init__(self, window_size=144):
        self.window_size = window_size

    def get_day(self, turbine_df: pd.DataFrame, selected_date: str):
        turbine_df = turbine_df.sort_values("time_stamp").copy()

        if "time_stamp" not in turbine_df.columns:
            raise ValueError("The dataframe must contain a 'time_stamp' column")

        day_start = pd.Timestamp(selected_date)
        day_end = day_start + pd.Timedelta(days=1)

        daily_df = turbine_df[(turbine_df["time_stamp"] >= day_start) & (turbine_df["time_stamp"] < day_end)]

        if daily_df.empty:
            return pd.DataFrame(columns=turbine_df.columns)

        return daily_df

    def get_latest_window(self, turbine_df: pd.DataFrame):
        turbine_df = turbine_df.sort_values("time_stamp")

        if len(turbine_df) < self.window_size:
            return turbine_df.copy()

        return turbine_df.tail(self.window_size).copy()