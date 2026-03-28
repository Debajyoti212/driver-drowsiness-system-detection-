"""
Module 05 (partial): SQLite database for driver profiles, sessions, calibration, and fatigue events.
"""

import sqlite3
import json
import time
from config import DB_PATH


class DriverDatabase:
    """Manages SQLite persistence for driver profiles and fatigue history."""

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drivers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                created_at REAL NOT NULL,
                total_sessions INTEGER DEFAULT 0,
                avg_drowsy_onset_min REAL DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calibration_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                driver_id INTEGER NOT NULL,
                ear_baseline REAL,
                mar_baseline REAL,
                ear_threshold REAL,
                mar_threshold REAL,
                pitch_baseline REAL,
                yaw_baseline REAL,
                calibrated_at REAL NOT NULL,
                FOREIGN KEY (driver_id) REFERENCES drivers(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                driver_id INTEGER NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL,
                duration_min REAL,
                max_drowsiness_score REAL DEFAULT 0,
                total_alerts INTEGER DEFAULT 0,
                total_yawns INTEGER DEFAULT 0,
                total_microsleeps INTEGER DEFAULT 0,
                avg_perclos REAL DEFAULT 0,
                FOREIGN KEY (driver_id) REFERENCES drivers(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fatigue_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                driver_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                details TEXT,
                drowsiness_score REAL,
                was_acknowledged INTEGER DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions(id),
                FOREIGN KEY (driver_id) REFERENCES drivers(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alert_dismissals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                alert_level TEXT NOT NULL,
                reason TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

        self.conn.commit()

    # ─── Driver Management ────────────────────────────────────────────────

    def create_driver(self, name):
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO drivers (name, created_at) VALUES (?, ?)",
                (name, time.time())
            )
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            row = cursor.execute(
                "SELECT id FROM drivers WHERE name = ?", (name,)
            ).fetchone()
            return row["id"]

    def get_driver(self, name):
        cursor = self.conn.cursor()
        row = cursor.execute(
            "SELECT * FROM drivers WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None

    def list_drivers(self):
        cursor = self.conn.cursor()
        rows = cursor.execute("SELECT * FROM drivers ORDER BY name").fetchall()
        return [dict(r) for r in rows]

    # ─── Calibration ──────────────────────────────────────────────────────

    def save_calibration(self, driver_id, ear_baseline, mar_baseline,
                         ear_threshold, mar_threshold,
                         pitch_baseline=0.0, yaw_baseline=0.0):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO calibration_data
            (driver_id, ear_baseline, mar_baseline, ear_threshold, mar_threshold,
             pitch_baseline, yaw_baseline, calibrated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (driver_id, ear_baseline, mar_baseline, ear_threshold, mar_threshold,
              pitch_baseline, yaw_baseline, time.time()))
        self.conn.commit()

    def get_latest_calibration(self, driver_id):
        cursor = self.conn.cursor()
        row = cursor.execute("""
            SELECT * FROM calibration_data
            WHERE driver_id = ?
            ORDER BY calibrated_at DESC LIMIT 1
        """, (driver_id,)).fetchone()
        return dict(row) if row else None

    # ─── Sessions ─────────────────────────────────────────────────────────

    def start_session(self, driver_id):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (driver_id, start_time) VALUES (?, ?)",
            (driver_id, time.time())
        )
        self.conn.commit()
        # Update driver session count
        cursor.execute(
            "UPDATE drivers SET total_sessions = total_sessions + 1 WHERE id = ?",
            (driver_id,)
        )
        self.conn.commit()
        return cursor.lastrowid

    def end_session(self, session_id, max_score, total_alerts,
                    total_yawns, total_microsleeps, avg_perclos):
        now = time.time()
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE sessions SET
                end_time = ?,
                duration_min = (? - start_time) / 60.0,
                max_drowsiness_score = ?,
                total_alerts = ?,
                total_yawns = ?,
                total_microsleeps = ?,
                avg_perclos = ?
            WHERE id = ?
        """, (now, now, max_score, total_alerts, total_yawns,
              total_microsleeps, avg_perclos, session_id))
        self.conn.commit()

    def get_driver_sessions(self, driver_id, limit=10):
        cursor = self.conn.cursor()
        rows = cursor.execute("""
            SELECT * FROM sessions WHERE driver_id = ?
            ORDER BY start_time DESC LIMIT ?
        """, (driver_id, limit)).fetchall()
        return [dict(r) for r in rows]

    # ─── Fatigue Events ───────────────────────────────────────────────────

    def log_fatigue_event(self, session_id, driver_id, event_type,
                          severity, details="", drowsiness_score=0.0):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO fatigue_events
            (session_id, driver_id, timestamp, event_type, severity, details, drowsiness_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_id, driver_id, time.time(), event_type, severity,
              details, drowsiness_score))
        self.conn.commit()

    def log_alert_dismissal(self, session_id, alert_level, reason=""):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO alert_dismissals (session_id, timestamp, alert_level, reason)
            VALUES (?, ?, ?, ?)
        """, (session_id, time.time(), alert_level, reason))
        self.conn.commit()

    def get_recent_dismissals(self, session_id, window_sec=600):
        cursor = self.conn.cursor()
        cutoff = time.time() - window_sec
        rows = cursor.execute("""
            SELECT * FROM alert_dismissals
            WHERE session_id = ? AND timestamp > ?
            ORDER BY timestamp DESC
        """, (session_id, cutoff)).fetchall()
        return [dict(r) for r in rows]

    def get_historical_patterns(self, driver_id):
        """Analyze historical fatigue patterns for a driver."""
        cursor = self.conn.cursor()
        sessions = cursor.execute("""
            SELECT duration_min, max_drowsiness_score, total_alerts
            FROM sessions WHERE driver_id = ? AND end_time IS NOT NULL
            ORDER BY start_time DESC LIMIT 50
        """, (driver_id,)).fetchall()

        if not sessions:
            return {"avg_drowsy_onset_min": 0, "avg_max_score": 0, "pattern": "insufficient_data"}

        durations = [s["duration_min"] or 0 for s in sessions]
        scores = [s["max_drowsiness_score"] or 0 for s in sessions]

        avg_duration = sum(durations) / len(durations) if durations else 0
        avg_score = sum(scores) / len(scores) if scores else 0

        pattern = "normal"
        if avg_score > 60:
            pattern = "frequently_drowsy"
        elif avg_duration > 120 and avg_score > 40:
            pattern = "long_drive_drowsy"

        return {
            "avg_drowsy_onset_min": avg_duration,
            "avg_max_score": avg_score,
            "pattern": pattern,
            "total_sessions": len(sessions),
        }

    def close(self):
        self.conn.close()
