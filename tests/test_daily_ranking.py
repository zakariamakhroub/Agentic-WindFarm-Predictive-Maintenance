import unittest

from agents.coordination_agent import CoordinationAgent
from agents.decision_agent import DecisionAgent
from agents.monitoring_agent import MonitoringAgent
from agents.prediction_agent import PredictionAgent
from preprocessing.data_cleaner import DataCleaner
from preprocessing.load_data import DataLoader


class DailyRankingWorkflowTests(unittest.TestCase):
    def test_selected_day_summary_and_ranking(self):
        loader = DataLoader("data/raw/datasets")
        cleaner = DataCleaner()
        monitoring_agent = MonitoringAgent()
        prediction_agent = PredictionAgent()
        decision_agent = DecisionAgent()
        coordination_agent = CoordinationAgent()

        turbine_df = cleaner.clean(loader.load_turbine("turbine1"))
        daily_df = monitoring_agent.get_day(turbine_df, "2023-09-25")

        self.assertFalse(daily_df.empty)

        prediction_results = prediction_agent.predict(daily_df)
        summary = prediction_agent.summarize_day(prediction_results, "2023-09-25")

        self.assertEqual(summary["date"], "2023-09-25")
        self.assertIn("mean_fault_probability", summary)
        self.assertIn("max_fault_probability", summary)

        decision = decision_agent.make_decision(summary)
        self.assertIn("risk_level", decision)
        self.assertIn("priority_score", decision)

        ranked = coordination_agent.prioritize([decision])
        self.assertEqual(ranked[0]["rank"], 1)

    def test_single_spike_does_not_become_critical(self):
        decision_agent = DecisionAgent()
        summary = {
            "mean_fault_probability": 0.05,
            "max_fault_probability": 0.95
        }

        decision = decision_agent.make_decision(summary)

        self.assertEqual(decision["risk_level"], "Medium")


if __name__ == "__main__":
    unittest.main()
