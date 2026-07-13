from preprocessing.load_data import DataLoader
from preprocessing.data_cleaner import DataCleaner
from preprocessing.data_splitter import DataSplitter
from preprocessing.target_creator import TargetCreator

from models.train_xgboost import XGBoostTrainer

from agents.monitoring_agent import MonitoringAgent
from agents.prediction_agent import PredictionAgent
from agents.decision_agent import DecisionAgent
from agents.memory_agent import MemoryAgent
from agents.coordination_agent import CoordinationAgent


# -------------------------------
# Configuration
# -------------------------------

selected_date = input("Enter the date (YYYY-MM-DD): ").strip() or "2022-09-25"
loader = DataLoader("data/raw/datasets")

cleaner = DataCleaner()
splitter = DataSplitter()
target_creator = TargetCreator()
trainer = XGBoostTrainer()

monitoring_agent = MonitoringAgent()
decision_agent = DecisionAgent()
memory_agent = MemoryAgent()
coordination_agent = CoordinationAgent()

all_decisions = []
model_trained = False
prediction_agent = None


# -------------------------------
# Process Every Turbine for the Selected Day
# -------------------------------

for turbine_name in loader.list_turbines():
    print(f"\nProcessing {turbine_name} for {selected_date}...")

    df = loader.load_turbine(turbine_name)
    df = cleaner.clean(df)

    daily_df = monitoring_agent.get_day(df, selected_date)
    if daily_df.empty:
        print(f"No data available for {turbine_name} on {selected_date}; skipping.")
        continue

    train_df, prediction_df = splitter.split(df)
    train_df = target_creator.create_target(train_df)
    prediction_df = target_creator.create_target(prediction_df)

    if not model_trained:
        trainer.train(train_df)
        trainer.save()
        prediction_agent = PredictionAgent()
        model_trained = True
        print("XGBoost model trained.")

    prediction_results = prediction_agent.predict(daily_df)
    summary = prediction_agent.summarize_day(prediction_results, selected_date)
    summary["turbine_id"] = turbine_name

    decision = decision_agent.make_decision(summary)
    memory_agent.store(decision)
    all_decisions.append(decision)


# -------------------------------
# Rank All Turbines for the Selected Day
# -------------------------------

ranked_schedule = coordination_agent.prioritize(all_decisions)

print("\n")
print("=" * 70)
print("MAINTENANCE SCHEDULE")
print(f"Date: {selected_date}")
print("=" * 70)

for item in ranked_schedule:
    print(
        f"Rank {item['rank']} | "
        f"{item['turbine_id']} | "
        f"Mean Probability: {item['mean_fault_probability']:.4f} | "
        f"Max Probability: {item['max_fault_probability']:.4f} | "
        f"Risk: {item['risk_level']} | "
        f"Action: {item['recommended_action']}"
    )