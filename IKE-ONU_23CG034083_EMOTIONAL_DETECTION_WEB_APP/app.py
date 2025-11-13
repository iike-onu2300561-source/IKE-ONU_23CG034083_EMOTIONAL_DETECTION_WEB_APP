from flask import Flask, render_template, request, jsonify
import os
import base64
import sqlite3
import datetime
import io
import json
from PIL import Image
import numpy as np
import cv2

# Import the refactored model class
from model import EmotionRecognizer  # renamed class from model.py

# ===========================
# Application Configuration
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_storage")
os.makedirs(DATA_DIR, exist_ok=True)

DB_FILE = os.path.join(DATA_DIR, "emotion_records.db")

# Initialize Flask App and Model
app = Flask(__name__)
emotion_model = EmotionRecognizer()


# ===========================
# Database Utility Functions
# ===========================
def get_db_connection():
    """Create and return a SQLite database connection."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    return conn


def record_prediction(emotion_label, probabilities):
    """Store prediction result into the local SQLite database."""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time_stamp TEXT,
                    emotion TEXT,
                    probabilities TEXT
                )"""
            )
            timestamp = datetime.datetime.utcnow().isoformat()
            cur.execute(
                "INSERT INTO results (time_stamp, emotion, probabilities) VALUES (?, ?, ?)",
                (timestamp, emotion_label, json.dumps(probabilities)),
            )
            conn.commit()
    except Exception as err:
        app.logger.error(f"[DB Error] Could not store prediction: {err}")


# ===========================
# Flask Routes
# ===========================
@app.route("/")
def home():
    """Render the main page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Receive an image (base64), perform emotion detection, and return results."""
    try:
        req_data = request.get_json(force=True)
        image_b64 = req_data.get("image")

        if not image_b64:
            return jsonify({"error": "Image data missing"}), 400

        # Decode base64 image
        header, encoded = image_b64.split(",", 1) if "," in image_b64 else ("", image_b64)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)

        # Run prediction
        emotion_label, probability_map = emotion_model.analyze_emotion(image_np)

        # Save results
        record_prediction(emotion_label, probability_map)

        return jsonify({"label": emotion_label, "probs": probability_map})

    except Exception as e:
        app.logger.error(f"[Server Error] {e}")
        return jsonify({"error": "Prediction failed"}), 500


# ===========================
# Server Entry Point
# ===========================
if __name__ == "__main__":
    print("🚀 Emotion Detection Web App running at: http://127.0.0.1:5000")
    app.run(debug=True)
