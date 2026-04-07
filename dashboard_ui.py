"""
Module 11: Performance Dashboard (UI)
- Live camera feed with landmark overlay, EAR/MAR values, alert level
- Real-time signal graphs (EAR, MAR, head pitch/yaw, drowsiness score) — 60s rolling
- Fatigue probability score gauge with confidence band
- False alarm feedback (thumbs up/down)
- System health panel (CPU, FPS, latency, signal confidence)
"""

import time
import threading
import numpy as np
from collections import deque

import cv2

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

import config


class DashboardUI:
    """
    OpenCV-based performance dashboard that renders alongside the main video feed.
    Uses cv2 drawing instead of Tkinter for simpler integration.
    """

    def __init__(self):
        # Graph data (60s rolling window at ~30fps)
        max_points = config.GRAPH_HISTORY_SEC * 30
        self.ear_history = deque(maxlen=max_points)
        self.mar_history = deque(maxlen=max_points)
        self.pitch_history = deque(maxlen=max_points)
        self.yaw_history = deque(maxlen=max_points)
        self.score_history = deque(maxlen=max_points)
        self.fps_history = deque(maxlen=max_points)

        # System metrics
        self.cpu_usage = 0.0
        self.ram_usage = 0.0
        self.last_system_update = 0

        # Feedback tracking
        self.false_alarm_count = 0
        self.true_alarm_count = 0
        self.feedback_log = deque(maxlen=200)

    def update_data(self, ear=0, mar=0, pitch=0, yaw=0, score=0, fps=0):
        """Add one frame's data to the rolling history."""
        self.ear_history.append(ear)
        self.mar_history.append(mar)
        self.pitch_history.append(pitch)
        self.yaw_history.append(yaw)
        self.score_history.append(score)
        self.fps_history.append(fps)

        # Update system metrics every 2 seconds
        now = time.time()
        if PSUTIL_AVAILABLE and now - self.last_system_update > 2:
            self.cpu_usage = psutil.cpu_percent()
            self.ram_usage = psutil.virtual_memory().percent
            self.last_system_update = now

    def log_feedback(self, is_true_alarm):
        """Log driver feedback on an alert (true alarm vs false alarm)."""
        if is_true_alarm:
            self.true_alarm_count += 1
        else:
            self.false_alarm_count += 1
        self.feedback_log.append({
            "time": time.time(),
            "true_alarm": is_true_alarm,
        })

    def render_dashboard(self, frame, eye_data, yawn_data, head_data,
                         score_data, alert_data, env_data,
                         occlusion_data, temporal_data, profile_data,
                         fps):
        """
        Render the complete dashboard overlay on the video frame.
        Returns a larger frame with dashboard panels.
        """
        h, w = frame.shape[:2]

        # Create dashboard canvas (wider to accommodate side panels)
        dash_w = w + 480  # extra space for right panel
        dash_h = max(h + 200, 700)  # extra space for bottom panel
        dashboard = np.zeros((dash_h, dash_w, 3), dtype=np.uint8)
        dashboard[:] = (30, 30, 30)  # dark background

        # ─── Place video feed ─────────────────────────────────────────
        dashboard[0:h, 0:w] = frame

        # ─── Top Status Bar ───────────────────────────────────────────
        self._draw_status_bar(dashboard, w, score_data, alert_data, env_data, profile_data)

        # ─── Right Panel: Signal Values ───────────────────────────────
        self._draw_signal_panel(dashboard, w, h, eye_data, yawn_data, head_data,
                                occlusion_data, temporal_data)

        # ─── Bottom Panel: Graphs ─────────────────────────────────────
        self._draw_graphs(dashboard, h, w)

        # ─── System Health (bottom right) ─────────────────────────────
        self._draw_system_health(dashboard, w, h, fps, occlusion_data)

        # ─── Alert Overlay on Video ───────────────────────────────────
        if alert_data and alert_data.get("triggered"):
            self._draw_alert_overlay(dashboard, w, h, alert_data)

        return dashboard

    def _draw_status_bar(self, canvas, video_w, score_data, alert_data,
                         env_data, profile_data):
        """Draw top status bar with score gauge and environment info."""
        bar_h = 5
        score = score_data.get("score", 0) if score_data else 0
        level = score_data.get("level", "none") if score_data else "none"

        # Color-coded score bar across top of video
        color = self._level_color(level)
        bar_width = int((score / 100.0) * video_w)
        cv2.rectangle(canvas, (0, 0), (bar_width, bar_h), color, -1)

    def _draw_signal_panel(self, canvas, x_start, video_h, eye_data, yawn_data,
                           head_data, occlusion_data, temporal_data):
        """Draw signal values panel on the right side."""
        x = x_start + 15
        y = 25
        line_h = 24
        section_gap = 15

        # Title
        cv2.putText(canvas, "SIGNAL MONITOR", (x, y),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += line_h + 5

        # ─── Eye Data ────────────────────────────────────────────────
        if eye_data:
            cv2.putText(canvas, "-- EYES --", (x, y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)
            y += line_h

            ear = eye_data.get("ear", 0)
            ear_color = (0, 255, 0) if ear > config.EAR_THRESHOLD else (0, 0, 255)
            cv2.putText(canvas, f"EAR: {ear:.3f}", (x, y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, ear_color, 1)
            y += line_h

            perclos = eye_data.get("perclos", 0)
            cv2.putText(canvas, f"PERCLOS: {perclos:.1%}", (x, y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y += line_h

            bpm = eye_data.get("blink_rate_bpm", 0)
            cv2.putText(canvas, f"Blinks/min: {bpm}", (x, y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y += line_h

            ms_count = eye_data.get("microsleep_count", 0)
            ms_color = (0, 0, 255) if ms_count > 0 else (255, 255, 255)
            cv2.putText(canvas, f"Microsleeps: {ms_count}", (x, y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, ms_color, 1)
            y += line_h

            if eye_data.get("sunglasses_detected"):
                cv2.putText(canvas, "SUNGLASSES DETECTED", (x, y),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                y += line_h

            y += section_gap

        # ─── Mouth Data ──────────────────────────────────────────────
        if yawn_data:
            cv2.putText(canvas, "-- MOUTH --", (x, y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)
            y += line_h

            mar = yawn_data.get("mar", 0)
            mar_color = (0, 0, 255) if yawn_data.get("mouth_open") else (0, 255, 0)
            cv2.putText(canvas, f"MAR: {mar:.3f}", (x, y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, mar_color, 1)
            y += line_h

            yawns = yawn_data.get("yawn_count", 0)
            freq = yawn_data.get("yawn_frequency_5min", 0)
            cv2.putText(canvas, f"Yawns: {yawns} (Freq: {freq}/5min)", (x, y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y += line_h

            if yawn_data.get("mouth_occluded"):
                cv2.putText(canvas, "MOUTH OCCLUDED", (x, y),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                y += line_h

            y += section_gap

        # ─── Head Pose ────────────────────────────────────────────────
        if head_data:
            cv2.putText(canvas, "-- HEAD POSE --", (x, y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)
            y += line_h

            pitch = head_data.get("pitch", 0)
            yaw = head_data.get("yaw", 0)
            roll = head_data.get("roll", 0)
            cv2.putText(canvas, f"Pitch: {pitch:.1f}  Yaw: {yaw:.1f}", (x, y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y += line_h

            cv2.putText(canvas, f"Roll: {roll:.1f}", (x, y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y += line_h

            mvmt = head_data.get("movement_rate", 0)
            cv2.putText(canvas, f"Movement: {mvmt:.1f} deg/s", (x, y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y += line_h

            if head_data.get("nod_event"):
                cv2.putText(canvas, "HEAD NOD!", (x, y),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                y += line_h

            y += section_gap

        # ─── Occlusion Status ────────────────────────────────────────
        if occlusion_data:
            state = occlusion_data.get("occlusion_state", "none")
            if state != "none":
                cv2.putText(canvas, "-- OCCLUSION --", (x, y),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
                y += line_h
                strategy = occlusion_data.get("fallback_strategy", "")
                # Word wrap for long strings
                words = strategy.split()
                line = ""
                for word in words:
                    if len(line + word) > 40:
                        cv2.putText(canvas, line, (x, y),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                        y += 18
                        line = word + " "
                    else:
                        line += word + " "
                if line:
                    cv2.putText(canvas, line.strip(), (x, y),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                    y += line_h

        # ─── Temporal Model ───────────────────────────────────────────
        if temporal_data:
            y += section_gap
            cv2.putText(canvas, "-- FATIGUE MODEL --", (x, y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)
            y += line_h

            pred = temporal_data.get("last_prediction", 0)
            conf = temporal_data.get("last_confidence", 0)
            using_fb = temporal_data.get("using_fallback", True)

            label = "Rule-based" if using_fb else "LSTM"
            cv2.putText(canvas, f"Prob: {pred:.1%} ({label})", (x, y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y += line_h

            cv2.putText(canvas, f"Confidence: {conf:.1%}", (x, y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    def _draw_graphs(self, canvas, video_h, video_w):
        """Draw mini signal graphs below the video feed."""
        graph_area_y = video_h + 10
        graph_h = 60
        graph_w = video_w // 2 - 20
        margin = 10

        graphs = [
            ("EAR", self.ear_history, (0, 255, 0), 0, 0.5),
            ("MAR", self.mar_history, (255, 200, 0), 1, 1.5),
            ("Score", self.score_history, (0, 100, 255), 2, 100),
            ("FPS", self.fps_history, (200, 200, 200), 3, 60),
        ]

        for name, data, color, idx, max_val in graphs:
            col = idx % 2
            row = idx // 2

            gx = margin + col * (graph_w + margin)
            gy = graph_area_y + row * (graph_h + 30)

            # Background
            cv2.rectangle(canvas, (gx, gy), (gx + graph_w, gy + graph_h),
                           (50, 50, 50), -1)
            cv2.rectangle(canvas, (gx, gy), (gx + graph_w, gy + graph_h),
                           (80, 80, 80), 1)

            # Label
            cv2.putText(canvas, name, (gx + 5, gy - 5),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            # Plot data
            if len(data) > 1:
                points = list(data)
                # Downsample for performance
                step = max(1, len(points) // graph_w)
                sampled = points[::step]

                for i in range(1, len(sampled)):
                    x1 = gx + int((i - 1) / len(sampled) * graph_w)
                    x2 = gx + int(i / len(sampled) * graph_w)

                    v1 = max(0, min(1, sampled[i - 1] / max_val if max_val > 0 else 0))
                    v2 = max(0, min(1, sampled[i] / max_val if max_val > 0 else 0))

                    y1 = gy + graph_h - int(v1 * graph_h)
                    y2 = gy + graph_h - int(v2 * graph_h)

                    cv2.line(canvas, (x1, y1), (x2, y2), color, 1)

                # Current value
                if sampled:
                    cv2.putText(canvas, f"{sampled[-1]:.2f}", (gx + graph_w - 60, gy + 15),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    def _draw_system_health(self, canvas, video_w, video_h, fps, occlusion_data):
        """Draw system health panel in bottom-right area."""
        x = video_w + 15
        y = video_h - 140 if video_h > 200 else video_h + 10

        cv2.putText(canvas, "-- SYSTEM HEALTH --", (x, y),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)
        y += 22

        # FPS
        fps_color = (0, 255, 0) if fps >= config.TARGET_FPS else (0, 0, 255)
        cv2.putText(canvas, f"FPS: {fps:.1f}", (x, y),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, fps_color, 1)
        y += 20

        # CPU/RAM
        if PSUTIL_AVAILABLE:
            cpu_color = (0, 255, 0) if self.cpu_usage < 80 else (0, 0, 255)
            cv2.putText(canvas, f"CPU: {self.cpu_usage:.0f}%", (x, y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, cpu_color, 1)
            y += 20

            cv2.putText(canvas, f"RAM: {self.ram_usage:.0f}%", (x, y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y += 20

        # Signal confidence
        if occlusion_data:
            signals = occlusion_data.get("available_signals", {})
            active = sum(1 for v in signals.values() if v)
            total = len(signals)
            cv2.putText(canvas, f"Signals: {active}/{total}", (x, y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y += 20

        # Feedback counts
        cv2.putText(canvas, f"Feedback: T:{self.true_alarm_count} F:{self.false_alarm_count}",
                     (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def _draw_alert_overlay(self, canvas, video_w, video_h, alert_data):
        """Draw alert overlay on the video feed area."""
        level = alert_data.get("level", "none")
        text = alert_data.get("display_text", "")
        color = self._level_color(level)

        if not text:
            return

        # Semi-transparent overlay at top of video
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 10), (video_w, 70), color, -1)
        cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)

        # Alert text
        cv2.putText(canvas, text, (15, 50),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Reasons (smaller text)
        reasons = alert_data.get("reasons", [])
        for i, reason in enumerate(reasons[:3]):  # show top 3 reasons
            y = 90 + i * 22
            cv2.putText(canvas, f"  > {reason}", (15, y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Rest recommendation
        rest = alert_data.get("rest_recommendation")
        if rest:
            y = 90 + len(reasons[:3]) * 22 + 10
            cv2.putText(canvas, rest, (15, y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1)

    def _draw_score_gauge(self, canvas, x, y, score, w=200, h=20):
        """Draw a horizontal score gauge."""
        # Background
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (60, 60, 60), -1)

        # Filled portion
        fill_w = int((score / 100.0) * w)
        if score < 30:
            color = (0, 200, 0)
        elif score < 55:
            color = (0, 255, 255)
        elif score < 75:
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)

        cv2.rectangle(canvas, (x, y), (x + fill_w, y + h), color, -1)

        # Border
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (120, 120, 120), 1)

        # Score text
        cv2.putText(canvas, f"{score:.0f}", (x + w + 10, y + h - 3),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    @staticmethod
    def _level_color(level):
        colors = {
            "none": (0, 180, 0),
            "low": (0, 255, 255),
            "medium": (0, 165, 255),
            "high": (0, 0, 255),
            "critical": (0, 0, 200),
        }
        return colors.get(level, (255, 255, 255))
