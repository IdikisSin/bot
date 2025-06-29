"""Utilities to load and save conversation history."""
import json
import logging
from .config import HISTORY_FILE


def load_history():
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as exc:
            logging.error("Error loading history: %s", exc)
            return []
    logging.info("No history file found. Starting with an empty history.")
    return []


def save_history(history):
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=4)
        logging.info("History saved successfully.")
    except Exception as exc:
        logging.error("Error saving history: %s", exc)
