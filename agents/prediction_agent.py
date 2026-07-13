import xgboost as xgb
import pandas as pd


class PredictionAgent:

    def __init__(self, model_path="models/xgboost_model.json"):

        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)

    def predict(self, turbine_df: pd.DataFrame):

        turbine_df = turbine_df.copy()

        excluded_columns = {
            "target",
            "status_type_id",
            "train_test"
        }

        feature_columns = [
            col
            for col in turbine_df.columns
            if col not in excluded_columns
            and pd.api.types.is_numeric_dtype(turbine_df[col])
        ]

        X = turbine_df[feature_columns]

        probabilities = self.model.predict_proba(X)[:, 1]

        predictions = self.model.predict(X)

        results = turbine_df.copy()

        results["fault_probability"] = probabilities
        results["prediction"] = predictions

        return results
    def summarize_window(self, prediction_results):

        summary = {

            "window_end":
                prediction_results["time_stamp"].iloc[-1],

            "max_fault_probability":
                prediction_results["fault_probability"].max(),

            "mean_fault_probability":
                prediction_results["fault_probability"].mean(),

            "predicted_faults":
                int(prediction_results["prediction"].sum())

        }

        return summary