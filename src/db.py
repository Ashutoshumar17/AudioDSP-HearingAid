"""
db.py
SQLite logging for processed audio experiments.
"""

import sqlite3

DB_PATH = "database/project.db"


def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS processed_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            samplerate INTEGER,
            filter_type TEXT,
            cutoff_low REAL,
            cutoff_high REAL,
            eq_gains TEXT,
            snr_before REAL,
            snr_after REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def log_result(filename, samplerate, filter_type, cutoff_low, cutoff_high,
                eq_gains, snr_before, snr_after, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO processed_files
        (filename, samplerate, filter_type, cutoff_low, cutoff_high, eq_gains, snr_before, snr_after)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (filename, samplerate, filter_type, cutoff_low, cutoff_high,
          str(eq_gains), snr_before, snr_after))
    conn.commit()
    conn.close()


def query_results(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM processed_files ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows


def query_best_improvements(db_path=DB_PATH, min_improvement=0.0):
    """Example of a more interesting SQL query: rank by SNR improvement."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT filename, snr_before, snr_after, (snr_after - snr_before) AS improvement
        FROM processed_files
        WHERE (snr_after - snr_before) > ?
        ORDER BY improvement DESC
    """, (min_improvement,))
    rows = cursor.fetchall()
    conn.close()
    return rows
