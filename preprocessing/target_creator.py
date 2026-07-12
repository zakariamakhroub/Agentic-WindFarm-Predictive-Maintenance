import pandas as pd


class TargetCreator:
    """
    Creates the target variable for XGBoost.

    Target:
        0 -> Normal
        1 -> Abnormal / Fault
    """

    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        normal_status = [0, 2]

        df["target"] = (~df["status_type_id"].isin(normal_status)).astype(int)

        return df