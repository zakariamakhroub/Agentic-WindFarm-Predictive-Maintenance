from agents.coordination_agent import CoordinationAgent
from preprocessing.data_cleaner import DataCleaner
from preprocessing.data_explorer import DataExplorer
from preprocessing.load_data import DataLoader
from preprocessing.data_splitter import DataSplitter
from preprocessing.target_creator import TargetCreator
from models.train_xgboost import XGBoostTrainer
from models.evaluator import ModelEvaluator
from agents.monitoring_agent import MonitoringAgent
from agents.decision_agent import DecisionAgent


import pandas as pd
loader = DataLoader("data/raw/datasets")

df = loader.load_turbine("turbine 0")


trainer = XGBoostTrainer()
evaluator = ModelEvaluator()

cleaner = DataCleaner()
splitter = DataSplitter()
target_creator = TargetCreator()


clean_df = cleaner.clean(df)

decision_agent = DecisionAgent()

train_df, prediction_df = splitter.split(clean_df)

train_df = target_creator.create_target(train_df)
prediction_df = target_creator.create_target(prediction_df)

model = trainer.train(train_df)

trainer.save()

evaluator.evaluate(model, prediction_df)

print("Model trained successfully.")

decision1 = decision

decision2 = {
    "turbine_id": "Turbine_2",
    "max_fault_probability": 0.73,
    "priority_score": 3,
    "risk_level": "High",
    "recommended_action": "Schedule Maintenance"
}

decision3 = {
    "turbine_id": "Turbine_3",
    "max_fault_probability": 0.18,
    "priority_score": 1,
    "risk_level": "Low",
    "recommended_action": "Continue Operation"
}

decision1["turbine_id"] = "Turbine_1"

coordination_agent = CoordinationAgent()

ranked = coordination_agent.prioritize([
    decision1,
    decision2,
    decision3
])

for turbine in ranked:
    print(turbine)

from agents.prediction_agent import PredictionAgent

prediction_agent = PredictionAgent()

monitoring_agent = MonitoringAgent(window_size=144)

window = monitoring_agent.get_latest_window(prediction_df)

prediction_results = prediction_agent.predict(window)

summary = prediction_agent.summarize_window(prediction_results)
decision = decision_agent.make_decision(summary)

from agents.memory_agent import MemoryAgent

memory_agent = MemoryAgent()










print(window["target"].value_counts())