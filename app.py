from functools import lru_cache
from io import BytesIO
import os

from flask import Flask, render_template, request, send_file, send_from_directory
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

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


@lru_cache(maxsize=64)
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
    navy = colors.HexColor("#0b3153")
    blue = colors.HexColor("#2563eb")
    muted = colors.HexColor("#475569")
    border = colors.HexColor("#dbe4ec")
    image_dir = os.path.join(app.root_path, "preprocessing", "image")

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=25,
        textColor=navy,
        alignment=1,
        spaceAfter=3 * mm,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=13,
        textColor=muted,
        alignment=1,
    )
    label_style = ParagraphStyle(
        "MetricLabel",
        parent=styles["Normal"],
        fontSize=7.5,
        leading=9,
        textColor=muted,
        alignment=1,
    )
    value_style = ParagraphStyle(
        "MetricValue",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=16,
        textColor=navy,
        alignment=1,
    )
    cell_style = ParagraphStyle(
        "Cell",
        parent=styles["Normal"],
        fontSize=7.5,
        leading=9,
        textColor=colors.HexColor("#243b53"),
    )
    header_cell_style = ParagraphStyle(
        "HeaderCell",
        parent=cell_style,
        fontName="Helvetica-Bold",
        textColor=colors.white,
    )

    left_logo = Image(os.path.join(image_dir, "emines.png"), width=43 * mm, height=15 * mm)
    right_logo = Image(os.path.join(image_dir, "uniten.png"), width=27 * mm, height=15 * mm)
    header = Table(
        [[left_logo, [Paragraph("Wind Farm Predictive Maintenance Report", title_style), Paragraph("Daily turbine risk and maintenance schedule", subtitle_style)], right_logo]],
        colWidths=[48 * mm, 104 * mm, 28 * mm],
    )
    header.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (1, 0), "CENTER"),
        ("ALIGN", (2, 0), (2, 0), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

    hero = Image(os.path.join(image_dir, "images234.png"), width=180 * mm, height=35 * mm)
    hero.hAlign = "CENTER"
    date_banner = Table(
        [[Paragraph(f"REPORT DATE  <b>{summary_metrics['selected_date']}</b>", ParagraphStyle("DateBanner", parent=cell_style, fontSize=9, textColor=colors.white))]],
        colWidths=[180 * mm],
    )
    date_banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), blue),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    metric_names = [("Turbines analyzed", "total_turbines"), ("Critical", "critical_count"), ("High", "high_count"), ("Medium", "medium_count"), ("Low", "low_count")]
    metric_cells = [[Paragraph(name, label_style), Paragraph(str(summary_metrics[key]), value_style)] for name, key in metric_names]
    metric_cells.append([Paragraph("Avg fault probability", label_style), Paragraph(f"{summary_metrics['avg_fault_probability']:.4f}", value_style)])
    metrics_table = Table([metric_cells], colWidths=[30 * mm] * 6)
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.7, border),
        ("INNERGRID", (0, 0), (-1, -1), 0.7, border),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))

    story = [
        header,
        Spacer(1, 5 * mm),
        hero,
        Spacer(1, 3 * mm),
        date_banner,
        Spacer(1, 5 * mm),
        metrics_table,
        Spacer(1, 8 * mm),
    ]

    schedule_data = [[Paragraph(label, header_cell_style) for label in ["Rank", "Turbine", "Mean probability", "Max probability", "Risk", "Action"]]]
    schedule_data.extend([
        [
            Paragraph(str(item["rank"]), cell_style),
            Paragraph(item["turbine_id"], cell_style),
            Paragraph(f"{item['mean_fault_probability']:.4f}", cell_style),
            Paragraph(f"{item['max_fault_probability']:.4f}", cell_style),
            Paragraph(item["risk_level"], cell_style),
            Paragraph(item["recommended_action"], cell_style),
        ]
        for item in ranked_schedule
    ])
    if len(schedule_data) == 1:
        schedule_data.append([Paragraph(value, cell_style) for value in ["-", "No data", "-", "-", "-", "No maintenance data available"]])

    schedule_table = Table(schedule_data, repeatRows=1, colWidths=[12 * mm, 22 * mm, 30 * mm, 30 * mm, 22 * mm, 59 * mm])
    schedule_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), navy),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, border),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f7")]),
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
