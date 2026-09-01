import pandas as pd

import app


def test_run_pipeline_reuses_cached_results_for_same_date(monkeypatch):
    calls = {"loader": 0}

    class FakeLoader:
        def __init__(self, data_path):
            calls["loader"] += 1

        def list_turbines(self):
            return ["turbine1"]

        def load_turbine(self, turbine_name):
            return pd.DataFrame(
                {
                    "time_stamp": ["2022-09-25 00:00:00", "2022-09-25 01:00:00"],
                    "temperature": [10.0, 11.0],
                    "power": [100.0, 110.0],
                    "target": [0, 1],
                }
            )

    class FakeCleaner:
        def clean(self, df):
            return df

    class FakeMonitoringAgent:
        def get_day(self, df, selected_date):
            return df

    class FakePredictionAgent:
        def predict(self, df):
            return pd.DataFrame(
                {
                    "time_stamp": df["time_stamp"],
                    "temperature": df["temperature"],
                    "fault_probability": [0.2, 0.7],
                    "prediction": [0, 1],
                }
            )

        def summarize_day(self, prediction_results, selected_date):
            return {
                "date": selected_date,
                "max_fault_probability": 0.7,
                "mean_fault_probability": 0.45,
                "predicted_faults": 1,
                "records_analyzed": 2,
            }

    class FakeDecisionAgent:
        def make_decision(self, summary):
            summary["risk_level"] = "High"
            summary["recommended_action"] = "Schedule Maintenance"
            summary["priority_score"] = 3
            summary["daily_priority_score"] = 0.45
            return summary

    class FakeMemoryAgent:
        def store(self, decision):
            return None

    class FakeCoordinationAgent:
        def prioritize(self, decisions):
            return [{
                "rank": 1,
                "turbine_id": "turbine1",
                "mean_fault_probability": 0.45,
                "max_fault_probability": 0.7,
                "risk_level": "High",
                "recommended_action": "Schedule Maintenance",
            }]

    monkeypatch.setattr(app, "DataLoader", FakeLoader)
    monkeypatch.setattr(app, "DataCleaner", FakeCleaner)
    monkeypatch.setattr(app, "MonitoringAgent", FakeMonitoringAgent)
    monkeypatch.setattr(app, "PredictionAgent", FakePredictionAgent)
    monkeypatch.setattr(app, "DecisionAgent", FakeDecisionAgent)
    monkeypatch.setattr(app, "MemoryAgent", FakeMemoryAgent)
    monkeypatch.setattr(app, "CoordinationAgent", FakeCoordinationAgent)

    app.run_pipeline.cache_clear()

    first = app.run_pipeline("2022-09-25")
    second = app.run_pipeline("2022-09-25")

    assert first == second
    assert calls["loader"] == 1
