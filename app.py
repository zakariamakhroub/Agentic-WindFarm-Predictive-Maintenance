from io import BytesIO

from flask import Flask, render_template, request, send_file, send_from_directory
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Spacer, Table, TableStyle, Paragraph

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


@app.route("/images/<path:filename>")
def images(filename):
    # Serve images placed in preprocessing/image (logos, turbine images)
    return send_from_directory("preprocessing/image", filename)


def run_pipeline(selected_date: str):
    loader = DataLoader("data/raw/datasets")
    cleaner = DataCleaner()
    monitoring_agent = MonitoringAgent()
    decision_agent = DecisionAgent()
    memory_agent = MemoryAgent()
    coordination_agent = CoordinationAgent()

    prediction_agent = PredictionAgent()
    all_decisions = []

    for turbine_name in loader.list_turbines():
        df = loader.load_turbine(turbine_name)
        df = cleaner.clean(df)

        daily_df = monitoring_agent.get_day(df, selected_date)
        if daily_df.empty:
            continue

        prediction_results = prediction_agent.predict(daily_df)
        summary = prediction_agent.summarize_day(prediction_results, selected_date)
        summary["turbine_id"] = turbine_name

        decision = decision_agent.make_decision(summary)
        memory_agent.store(decision)
        all_decisions.append(decision)

    ranked_schedule = coordination_agent.prioritize(all_decisions)
    total_turbines = len(all_decisions)
    avg_fault_probability = (
        sum(item["mean_fault_probability"] for item in all_decisions) / total_turbines
        if total_turbines > 0
        else 0.0
    )
    summary_metrics = {
        "selected_date": selected_date,
        "total_turbines": total_turbines,
        "critical_count": sum(1 for item in all_decisions if item.get("risk_level") == "Critical"),
        "high_count": sum(1 for item in all_decisions if item.get("risk_level") == "High"),
        "medium_count": sum(1 for item in all_decisions if item.get("risk_level") == "Medium"),
        "low_count": sum(1 for item in all_decisions if item.get("risk_level") == "Low"),
        "avg_fault_probability": round(avg_fault_probability, 4),
    }

    return ranked_schedule, summary_metrics


@app.route("/download-report", methods=["GET"])
def download_report():
    selected_date = request.args.get("selected_date", "2022-09-25").strip() or "2022-09-25"
    ranked_schedule, summary_metrics = run_pipeline(selected_date)

    report_buffer = BytesIO()
    document = SimpleDocTemplate(
        report_buffer,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Wind Farm Predictive Maintenance Report", styles["Title"]),
        Paragraph(f"Selected date: {summary_metrics['selected_date']}", styles["Normal"]),
        Spacer(1, 8 * mm),
    ]

    metrics_data = [
        ["Turbines analyzed", "Critical", "High", "Medium", "Low", "Avg fault probability"],
        [
            str(summary_metrics["total_turbines"]),
            str(summary_metrics["critical_count"]),
            str(summary_metrics["high_count"]),
            str(summary_metrics["medium_count"]),
            str(summary_metrics["low_count"]),
            f"{summary_metrics['avg_fault_probability']:.4f}",
        ],
    ]
    metrics_table = Table(metrics_data, repeatRows=1)
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b8c7d3")),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#edf3f6")),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 7),
        ("TOPPADDING", (0, 0), (-1, 0), 7),
    ]))
    story.extend([metrics_table, Spacer(1, 8 * mm)])

    schedule_data = [["Rank", "Turbine", "Mean probability", "Max probability", "Risk", "Action"]]
    schedule_data.extend([
        [
            str(item["rank"]),
            item["turbine_id"],
            f"{item['mean_fault_probability']:.4f}",
            f"{item['max_fault_probability']:.4f}",
            item["risk_level"],
            item["recommended_action"],
        ]
        for item in ranked_schedule
    ])
    if len(schedule_data) == 1:
        schedule_data.append(["-", "No data", "-", "-", "-", "No maintenance data available"])

    schedule_table = Table(schedule_data, repeatRows=1, colWidths=[12 * mm, 22 * mm, 30 * mm, 30 * mm, 22 * mm, 59 * mm])
    schedule_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b8c7d3")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafb")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f7")]),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(schedule_table)
    document.build(story)
    report_buffer.seek(0)

    return send_file(
        report_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"wind-farm-report-{selected_date}.pdf",
    )


@app.route("/", methods=["GET", "POST"])
def index():
    selected_date = "2022-09-25"
    error_message = None
    ranked_schedule = []
    summary_metrics = {
        "selected_date": selected_date,
        "total_turbines": 0,
        "critical_count": 0,
        "high_count": 0,
        "medium_count": 0,
        "low_count": 0,
        "avg_fault_probability": 0.0,
    }

    if request.method == "POST":
        selected_date = request.form.get("selected_date", selected_date).strip() or selected_date

    try:
        ranked_schedule, summary_metrics = run_pipeline(selected_date)
    except Exception as exc:
        error_message = f"Unable to generate the dashboard: {exc}"

    return render_template(
        "index.html",
        selected_date=selected_date,
        ranked_schedule=ranked_schedule,
        summary_metrics=summary_metrics,
        error_message=error_message,
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
