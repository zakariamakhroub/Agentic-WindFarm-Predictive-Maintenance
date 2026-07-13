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
# Initialize Components
# -------------------------------

loader = DataLoader("data/raw/datasets")

cleaner = DataCleaner()

splitter = DataSplitter()

target_creator = TargetCreator()

trainer = XGBoostTrainer()

monitoring_agent = MonitoringAgent(window_size=144)

decision_agent = DecisionAgent()

memory_agent = MemoryAgent()

coordination_agent = CoordinationAgent()

all_decisions = []

model_trained = False

prediction_agent = None


# -------------------------------
# Process Every Turbine
# -------------------------------

for turbine_name in loader.list_turbines():

    print(f"\nProcessing {turbine_name}...")

    # Load
    df = loader.load_turbine(turbine_name)

    # Clean
    df = cleaner.clean(df)

    # Split
    train_df, prediction_df = splitter.split(df)

    # Create target
    train_df = target_creator.create_target(train_df)
    prediction_df = target_creator.create_target(prediction_df)

    # Train the model only once
    if not model_trained:

        model = trainer.train(train_df)

        trainer.save()

        prediction_agent = PredictionAgent()

        model_trained = True

        print("XGBoost model trained.")

    # Monitoring
    window = monitoring_agent.get_latest_window(prediction_df)

    # Prediction
    prediction_results = prediction_agent.predict(window)

    summary = prediction_agent.summarize_window(prediction_results)

    # Add turbine ID
    summary["turbine_id"] = turbine_name

    # Decision
    decision = decision_agent.make_decision(summary)

    # Memory
    memory_agent.store(decision)

    # Save for ranking
    all_decisions.append(decision)


# -------------------------------
# Rank All Turbines
# -------------------------------

ranked_schedule = coordination_agent.prioritize(all_decisions)

print("\n")
print("=" * 70)
print("RANKED MAINTENANCE SCHEDULE")
print("=" * 70)

for item in ranked_schedule:

    print(
        f"Rank {item['rank']} | "
        f"{item['turbine_id']} | "
        f"Probability: {item['mean_fault_probability']:.4f} | "
        f"Risk: {item['risk_level']} | "
        f"Action: {item['recommended_action']}"
    )