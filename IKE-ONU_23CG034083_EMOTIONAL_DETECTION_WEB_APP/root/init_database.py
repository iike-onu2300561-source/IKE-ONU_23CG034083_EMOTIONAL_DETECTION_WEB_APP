import sqlite3
import os

# Path setup (match your app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_storage")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "emotion_records.db")

def initialize_db():
    """Create the database and table for emotion predictions."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time_stamp TEXT NOT NULL,
            emotion TEXT NOT NULL,
            probabilities TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print(f"✅ Database initialized successfully at: {DB_PATH}")

if __name__ == "__main__":
    initialize_db()
