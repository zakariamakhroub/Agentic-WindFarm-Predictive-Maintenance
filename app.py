from flask import Flask, render_template, request

from preprocessing.load_data import DataLoader
from preprocessing.data_cleaner import DataCleaner
from preprocessing.data_splitter import DataSplitter
from preprocessing.target_creator import TargetCreator

from agents.monitoring_agent import MonitoringAgent
from agents.prediction_agent import PredictionAgent
from agents.decision_agent import DecisionAgent
from agents.memory_agent import MemoryAgent
from agents.coordination_agent import CoordinationAgent

app = Flask(__name__)


def run_pipeline(selected_date: str):
    loader = DataLoader("data/raw/datasets")
    cleaner = DataCleaner()
    splitter = DataSplitter()
    target_creator = TargetCreator()

    monitoring_agent = MonitoringAgent()
    decision_agent = DecisionAgent()
    memory_agent = MemoryAgent()
    coordination_agent = CoordinationAgent()

    all_decisions = []

    for turbine_name in loader.list_turbines():
        df = loader.load_turbine(turbine_name)
        df = cleaner.clean(df)

        daily_df = monitoring_agent.get_day(df, selected_date)
        if daily_df.empty:
            continue

        train_df, _ = splitter.split(df)
        train_df = target_creator.create_target(train_df)

        prediction_agent = PredictionAgent()
        prediction_results = prediction_agent.predict(daily_df)
        summary = prediction_agent.summarize_day(prediction_results, selected_date)
        summary["turbine_id"] = turbine_name

        decision = decision_agent.make_decision(summary)
        memory_agent.store(decision)
        all_decisions.append(decision)

    ranked_schedule = coordination_agent.prioritize(all_decisions)
    return ranked_schedule


@app.route("/", methods=["GET", "POST"])
def index():
    selected_date = "2022-09-25"
    error_message = None
    ranked_schedule = []

    if request.method == "POST":
        selected_date = request.form.get("selected_date", selected_date).strip() or selected_date

    try:
        ranked_schedule = run_pipeline(selected_date)
    except Exception as exc:
        error_message = f"Unable to generate the dashboard: {exc}"

    return render_template(
        "index.html",
        selected_date=selected_date,
        ranked_schedule=ranked_schedule,
        error_message=error_message,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
