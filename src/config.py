from pathlib import Path
import os
import tomllib  # Python 3.11+

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Pilih config via env (default: config.toml)
CONFIG_PATH = os.getenv(
    "APP_CONFIG",
    PROJECT_ROOT / "config" / "config.toml"
)

with open(CONFIG_PATH, "rb") as f:
    CONFIG = tomllib.load(f)

# Paths
DATA_DIR = PROJECT_ROOT / CONFIG["paths"]["data_dir"]
RAW_DIR = PROJECT_ROOT / CONFIG["paths"]["raw_dir"]
PROCESS_DIR = PROJECT_ROOT / CONFIG["paths"]["process_dir"]
CLEAN_DIR = PROJECT_ROOT / CONFIG["paths"]["clean_dir"]
MODEL_DIR = PROJECT_ROOT / CONFIG["paths"]["models_dir"]

# API
API_HOST = CONFIG["api"]["host"]
API_PORT = CONFIG["api"]["port"]

# Weather / features
TIMEZONE = CONFIG["weather"]["timezone"]
ROLLING_WINDOWS = CONFIG["features"]["rolling_windows"]
