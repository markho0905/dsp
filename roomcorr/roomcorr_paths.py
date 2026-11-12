"""roomcorr_paths.py - shared locations (kept out of the entry script so frozen builds can import them)."""
import os
import sys

APP_DIR = os.path.expanduser("~/Library/Application Support/RoomCorr") if sys.platform == "darwin" \
    else os.path.expanduser("~/.roomcorr")
PROFILE_DIR = os.path.join(APP_DIR, "profiles")
SESS_DIR = os.path.join(APP_DIR, "sessions")
LOG_FILE = os.path.expanduser("~/Library/Logs/RoomCorr.log") if sys.platform == "darwin" \
    else os.path.join(APP_DIR, "RoomCorr.log")
