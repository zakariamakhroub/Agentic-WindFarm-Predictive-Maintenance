import pandas as pd


class MonitoringAgent:
    """
    Selects the most recent SCADA observations for analysis.
    """

    def __init__(self, window_size=144):
        self.window_size = window_size

    def get_latest_window(self, turbine_df: pd.DataFrame):

        turbine_df = turbine_df.sort_values("time_stamp")

        if len(turbine_df) < self.window_size:
            return turbine_df.copy()

        return turbine_df.tail(self.window_size).copy()