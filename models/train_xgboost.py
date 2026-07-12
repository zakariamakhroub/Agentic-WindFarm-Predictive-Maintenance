import xgboost as xgb
import pandas as pd


class XGBoostTrainer:

    def __init__(self):

        self.model = xgb.XGBClassifier(

            objective="binary:logistic",

            eval_metric="logloss",

            n_estimators=100,

            learning_rate=0.1,

            max_depth=6,

            random_state=42,

            scale_pos_weight=11.4
        )

    def train(self, train_df: pd.DataFrame):

        train_df = train_df.copy()

        excluded_columns = {"target", "status_type_id", "train_test"}

        feature_columns = [
            col
            for col in train_df.columns
            if col not in excluded_columns
            and pd.api.types.is_numeric_dtype(train_df[col])
        ]

        X = train_df[feature_columns]
        y = train_df["target"]

        self.model.fit(X, y)

        return self.model

    def save(self, path="models/xgboost_model.json"):

        self.model.save_model(path)