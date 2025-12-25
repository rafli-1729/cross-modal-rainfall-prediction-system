import streamlit as st
import requests
from datetime import timedelta, datetime
import pandas as pd
import altair as alt

from utils import (
    today_sg,
    set_bg,
    render_template,
    load_css,
    evaluate_prediction,
    rainfall_intensity,
    forecast_insight_text,
    scenario_insight_text
)

from src.config import VALID_LOCATIONS

@st.cache_data(show_spinner=True)
def load_raw_data():
    df = pd.read_csv("data/clean/train.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(show_spinner=True)
def build_weekly_aggregates(df):
    return (
        df
        .assign(year=df["date"].dt.year)
        .set_index("date")
        .groupby(["location", "year"])
        .resample("W")
        .mean()
        .reset_index()
    )


# =============================== CONFIG ===============================

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Rainfall Forecasting App",
    layout="wide",
)

# set_bg("assets/bg.webp")
load_css("assets/styles.css")

# =============================== HEADER ===============================

st.title("🌧️ Rainfall Forecasting Dashboard")
st.caption("Monitoring rainfall patterns and interactively evaluating prediction models.")

st.divider()

# =============================== DASHBOARD (DUMMY) ===============================

st.subheader("Overview")

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Locations Covered", "25")
kpi2.metric("Evaluation Records", "12,840")
kpi3.metric("Model Type", "XGBoost")

st.markdown("### Recent Rainfall Trend (Dummy)")

df = pd.DataFrame({
    "Day": range(7),
    "Observed": [2, 5, 0, 10, 3, 6, 1],
    "Predicted": [1, 4, 2, 7, 4, 5, 2]
}).melt("Day", var_name="Type", value_name="Rainfall")

with st.spinner("Initializing dashboard..."):
    df = load_raw_data()
    weekly = build_weekly_aggregates(df)

available_locations = sorted(weekly["location"].unique())
available_years = sorted(weekly["year"].unique())

col1, col2 = st.columns(2)
with col1:
    selected_location = st.selectbox(
        "Location",
        available_locations
    )

with col2:
    selected_year = st.selectbox(
        "Year",
        available_years,
        index=len(available_years) - 1
    )

filtered = weekly[
    (weekly["location"] == selected_location) &
    (weekly["year"] == selected_year)
]

if filtered.empty:
    st.info("No data available for this selection.")
    st.stop()

plot_df = filtered.melt(
    id_vars=["date"],
    value_vars=["observed_mm", "predicted_mm"],
    var_name="Type",
    value_name="Rainfall"
)

plot_df["Type"] = plot_df["Type"].map({
    "observed_mm": "Observed",
    "predicted_mm": "Predicted"
})

chart = (
    alt.Chart(plot_df)
    .mark_line(interpolate="monotone", strokeWidth=2)
    .encode(
        x=alt.X(
            "date:T",
            axis=alt.Axis(
                title=None,
                labelColor="#64748b"
            )
        ),
        y=alt.Y(
            "Rainfall:Q",
            axis=alt.Axis(
                title="mm",
                labelColor="#64748b",
                gridColor="#e5e7eb",
                gridOpacity=0.6
            )
        ),
        color=alt.Color(
            "Type:N",
            scale=alt.Scale(
                domain=["Observed", "Predicted"],
                range=["#0f172a", "#60a5fa"]
            ),
            legend=alt.Legend(
                orient="bottom",
                title=None
            )
        ),
        strokeDash=alt.condition(
            alt.datum.Type == "Predicted",
            alt.value([4, 4]),
            alt.value([1, 0])
        )
    )
    .properties(height=260)
    .configure_view(strokeWidth=0)
    .configure(background="transparent")
)

st.altair_chart(chart, use_container_width=True)

st.divider()

# =============================== TRY THE MODEL (BOTTOM SECTION) ===============================

st.markdown("## Try the Model")
st.caption("Test the rainfall prediction model using different modes and inputs.")

mode = st.selectbox(
    "",
    ["Evaluation", "Forecast", "Random Scenario"]
)

# ================================== COMMON INPUTS ==================================

col1, col2 = st.columns(2)
with col1:
    location = st.selectbox("Location", VALID_LOCATIONS)

with col2:
    date = st.date_input("Date")

date_str = date.strftime("%Y-%m-%d")
date_formatted = date.strftime("%d %b %Y")

# ================================== FORECAST TAB ==================================

if mode == 'Random Scenario':
    st.subheader("Scenario Inputs")

    col1, col2, col3 = st.columns(3)

    with col1:
        r30 = st.number_input("Highest 30-min Rainfall (mm)", 0.0, 200.0, 1.0, step=0.5)
        min_temp = st.number_input("Min Temperature (°C)", 0.0, 40.0, 24.0)

    with col2:
        r60 = st.number_input("Highest 60-min Rainfall (mm)", 0.0, 300.0, 1.0, step=0.5)
        max_temp = st.number_input("Max Temperature (°C)", 0.0, 40.0, 30.0)

    with col3:
        r120 = st.number_input("Highest 120-min Rainfall (mm)", 0.0, 500.0, 1.0, step=0.5)
        mean_temp = st.number_input("Mean Temperature (°C)", 0.0, 40.0, 27.0)

    col1, col2 = st.columns(2)
    with col1:
        mean_wind = st.number_input("Mean Wind Speed (km/h)", 0.0, 50.0, 8.0)
    with col2:
        max_wind = st.number_input("Max Wind Speed (km/h)", 0.0, 100.0, 15.0)

    if st.button("Run Scenario", use_container_width=True, type='primary'):
        payload = {
            "features": {
                "location": location,
                "date": date_str,
                "mean_temperature_c": mean_temp,
                "maximum_temperature_c": max_temp,
                "minimum_temperature_c": min_temp,
                "mean_wind_speed_kmh": mean_wind,
                "max_wind_speed_kmh": max_wind,
                "highest_30_min_rainfall_mm": r30,
                "highest_60_min_rainfall_mm": r60,
                "highest_120_min_rainfall_mm": r120,
            }
        }

        result_slot = st.empty()
        with result_slot:
            render_template("assets/templates/loading.html")

        res = requests.post(f"{API_BASE}/random", json=payload)

        data = res.json()
        rain_mm = float(data["prediction"]["daily_rainfall_mm"])

        level, label = rainfall_intensity(rain_mm)
        insight = scenario_insight_text(rain_mm, level)

        scenario_inputs = {
            "Feature Source": data["meta"]["feature_source"],
        }

        with result_slot:
            render_template(
                "assets/templates/random_result.html",
                rainfall_mm=f"{rain_mm:.1f}",
                intensity_level=level,
                intensity_label=label,
                location=data["input"]["location"]
            )

# ================================= FORECAST MODE =================================

if mode == 'Forecast':

    is_valid = True
    error_msg = None

    today = today_sg()
    max_forecast = today + timedelta(days=14)

    if date < today:
        is_valid = False
        error_msg = "Forecast date cannot be in the past."

    elif date > max_forecast:
        is_valid = False
        error_msg = (
            f"Forecast only available up to {max_forecast}"
        )

    if not is_valid:
        st.warning(error_msg)

    if st.button("Forecast", disabled=not is_valid, use_container_width=True, type='primary'):
        payload = {
            "location": location,
            "date": date_str
        }

        result_slot = st.empty()
        with result_slot:
            render_template("assets/templates/loading.html")

        res = requests.post(f"{API_BASE}/forecast", json=payload)

        data = res.json()
        rain_mm = float(data["prediction"]["daily_rainfall_mm"])

        level, label = rainfall_intensity(rain_mm)
        insight = forecast_insight_text(rain_mm, level)

        with result_slot:
            render_template(
                "assets/templates/forecast_result.html",
                rainfall_mm=f"{rain_mm:.1f}",
                intensity_level=level,
                intensity_label=label,
                insight_text=insight,
                location=data["input"]["location"],
                date=date_formatted
            )


# ================================= EVALUATION MODE =================================

if mode == 'Evaluation':
    is_valid = True
    error_msg = None

    today = today_sg()
    selected_date = date

    if selected_date >= today:
        is_valid = False
        error_msg = (
            "Evaluation requires observed data. "
            "Please select a date before today (SGT)."
        )

    if not is_valid:
        st.warning(error_msg)

    if st.button("Run Evaluation", disabled=not is_valid, use_container_width=True, type='primary'):
        payload = {
            "location": location,
            "date": selected_date.strftime("%Y-%m-%d")
        }

        result_slot = st.empty()
        with result_slot:
            render_template("assets/templates/loading.html")

        res = requests.post(f"{API_BASE}/evaluate", json=payload)
        data = res.json()

        mode = data["mode"]
        location = data["input"]["location"]
        date = data["input"]["date"]
        pred_mm = data["prediction"]["daily_rainfall_mm"]
        obs_mm = data["comparison"]["observed_daily_rainfall_mm"]
        error_mm = data["comparison"]["error_mm"]

        with result_slot:
            if data["comparison"] is None:
                render_template(
                    "assets/templates/evaluation_unavailable.html",
                    location=location,
                    date=date
                )
            else:
                eval_result = evaluate_prediction(pred_mm, obs_mm)
                render_template(
                    "assets/templates/evaluate_result.html",
                    rainfall_mm=f"{eval_result['predicted_mm']:.1f}",
                    obs_rainfall_mm=f"{eval_result['observed_mm']:.1f}",
                    error_pct=f"{eval_result['relative_error_pct']:.1f}",
                    error_level=eval_result["severity"],
                    insight_text=eval_result["insight_text"],
                    location=location,
                    date=date_formatted
                )

render_template("assets/footer.html")