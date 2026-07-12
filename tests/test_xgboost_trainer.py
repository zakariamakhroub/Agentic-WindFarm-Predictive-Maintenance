import pandas as pd

from models.train_xgboost import XGBoostTrainer


class DummyModel:
    def __init__(self, *args, **kwargs):
        self.fit_calls = []

    def fit(self, X, y):
        self.fit_calls.append((X, y))
        return self


def test_train_drops_datetime_columns_before_fit(monkeypatch):
    captured = {}

    class DummyClassifier(DummyModel):
        pass

    def factory(*args, **kwargs):
        model = DummyClassifier(*args, **kwargs)
        captured["model"] = model
        return model

    monkeypatch.setattr("models.train_xgboost.xgb.XGBClassifier", factory)

    trainer = XGBoostTrainer()
    df = pd.DataFrame(
        {
            "time_stamp": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
            "target": [0, 1],
            "status_type_id": [1, 1],
            "train_test": [0, 0],
            "feature": [1.5, 2.5],
        }
    )

    trainer.train(df)

    X, y = captured["model"].fit_calls[0]
    assert "time_stamp" not in X.columns
    assert list(X.columns) == ["feature"]
    assert y.tolist() == [0, 1]
