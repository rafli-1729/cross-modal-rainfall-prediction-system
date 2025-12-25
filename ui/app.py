import streamlit as st
import requests
import json
import base64

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import VALID_LOCATIONS

API_BASE = "http://localhost:8000"

def today_sg():
    return datetime.now(ZoneInfo("Asia/Singapore")).date()

def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg("assets/bg.webp")

def load_css(path: str):
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("assets/styles.css")

st.set_page_config(
    page_title="Rainfall Forecasting App",
    layout="centered"
)

st.title("🌧️ Rainfall Forecasting & Evaluation")

# ================================= MODE SELECTION =================================

mode = st.selectbox(
    "Select Mode",
    ["Random Prediction", "Forecast", "Evaluation"]
)

st.divider()

# ================================= COMMON INPUTS =================================

col1, col2= st.columns(2)

with col1:
    location = st.text_input("Location", value="Admiralty")
    location = location.replace(" ", "_").title()

    is_valid_location = location in VALID_LOCATIONS

    if not is_valid_location:
        st.error("Invalid location. Please select a valid location.")

with col2:
    date = st.date_input("Date")

date_str = date.strftime("%Y-%m-%d")

# ================================= RANDOM MODE =================================

if mode == "Random Prediction":
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

    if st.button("Run Scenario", use_container_width=True):
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

        res = requests.post(f"{API_BASE}/random", json=payload)
        st.write(f"Status code: {res.status_code}")

        try:
            st.json(res.json())
        except Exception:
            st.error("API did not return valid JSON")
            st.subheader("Raw response:")
            st.code(res.text)

# ================================= FORECAST MODE =================================

elif mode == "Forecast":

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

    if st.button("Run Forecast", use_container_width=True):
        payload = {
            "location": location,
            "date": date_str
        }
        res = requests.post(f"{API_BASE}/forecast", json=payload)
        st.write(f"Status code: {res.status_code}")

        try:
            st.json(res.json())
        except Exception:
            st.error("API did not return valid JSON")
            st.subheader("Raw response:")
            st.code(res.text)

# ================================= EVALUATION MODE =================================

elif mode == "Evaluation":

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

    if st.button("Run Evaluation", disabled=not is_valid):
        payload = {
            "location": location,
            "date": selected_date.strftime("%Y-%m-%d")
        }

        res = requests.post(f"{API_BASE}/evaluate", json=payload)

        st.write(f"Status code: {res.status_code}")
        try:
            st.json(res.json())
        except Exception:
            st.error("API did not return valid JSON")
            st.code(res.text)