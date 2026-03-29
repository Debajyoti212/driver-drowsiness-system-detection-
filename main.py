"""
Driver Drowsiness Detection System - Main Entry Point
=====================================================
Orchestrates all 14 modules:
  01. Real-Time Video Processing
  02. Eye State Detection (EAR, PERCLOS, Blink, Microsleep, Gaze)
  03. Mouth / Yawn Detection (MAR, frequency, long-duration)
  04. Head Pose Estimation (PnP, nod, lean, movement rate)
  05. Driver Identity & Personalization (profiles, calibration, SQLite)
  06. Environmental Context Awareness (time-of-day, trip duration, lighting)
  07. Temporal Fatigue Analysis (LSTM/GRU or rule-based fallback)
  08. Multi-Signal Drowsiness Scoring Engine (weighted, confidence-gated)
  09. Intelligent Alert System (graded, escalation, explainability)
  10. Occlusion & Edge Case Handling (sunglasses, partial face, fail-safe)
  11. Performance Dashboard (signals, graphs, system health)
  12. Analytics, Logging & Reports (CSV/JSON/PDF, FP rate, ROC/AUC)
  13. Cross-Platform Deployment (ONNX, TensorRT, Docker)
  14. Documentation & Research Standards (auto-gen architecture docs)

Controls:
  ESC         - Quit
  C           - Start calibration
  D           - Toggle dashboard
  A           - Acknowledge alert
  X           - Dismiss alert (false alarm)
  F           - Feedback: mark last alert as true alarm
  R           - Export session report (CSV + JSON)
  G           - Generate documentation
  1-9         - Select driver profile (creates if needed)
"""

import sys
import os
import cv2
import time
import threading
import http.server
import webbrowser
from flask import Response

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from modules.video_processor import VideoProcessor
from modules.eye_detector import EyeDetector
from modules.yawn_detector import YawnDetector
from modules.head_pose import HeadPoseEstimator
from modules.driver_profile import DriverProfile
from modules.environment import EnvironmentAwareness
from modules.temporal_model import TemporalFatigueModel
from modules.scoring_engine import DrowsinessScorer
from modules.alert_system import AlertSystem
from modules.occlusion_handler import OcclusionHandler
from modules.analytics import SessionLogger
from modules.documentation import DocumentationGenerator
from dashboard.dashboard_ui import DashboardUI
# ================= FLASK SETUP =================
from flask import Flask, jsonify

app = Flask(__name__)

latest_data = {
    
    "ear": 0,
    "mar": 0,
    "score": 0,
    "fatigue": 0
}
camera_running = False
video = None
# ==============================================
@app.route('/data')
def get_data():
    return jsonify(latest_data)


# ✅ START CAMERA ROUTE
@app.route('/start')
def start_camera():
    global camera_running

    if not camera_running:
        camera_running = True
        threading.Thread(target=main, daemon=True).start()

    return "Camera Started"


# ✅ STOP CAMERA ROUTE
@app.route('/stop')
def stop_camera():
    global camera_running
    camera_running = False
    return "Camera Stopped"


# 🎥 VIDEO STREAM
def generate_frames():
    global video, camera_running

    while camera_running:
        ret, frame, fps = video.read_frame()
        if not ret:
            continue

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# 👇 EXISTING CODE CONTINUES
def start_web_dashboard():
    app.run(port=5000)

from flask import render_template

@app.route('/')
def home():
    return render_template("index.html")

def start_web_dashboard():
    app.run(port=5000)
def main():
    global video
    print("=" * 60)
    print("  DRIVER DROWSINESS DETECTION SYSTEM")
    video = VideoProcessor()
    # ─── Start Web Dashboard Server ─────────────
    web_thread = threading.Thread(target=start_web_dashboard, daemon=True)
    web_thread.start()

    print("  Web dashboard running at http://localhost:5000")
    webbrowser.open("http://localhost:5000")
    print("  14-Module Comprehensive Safety Monitor")
    print("=" * 60)

    # ─── Initialize all modules ───────────────────────────────────────
    video = VideoProcessor()
    eyes = EyeDetector()
    yawn = YawnDetector()
    head = HeadPoseEstimator()
    profile = DriverProfile()
    environment = EnvironmentAwareness()
    temporal = TemporalFatigueModel()
    scorer = DrowsinessScorer()
    alerts = AlertSystem()
    occlusion = OcclusionHandler()
    logger = SessionLogger()
    doc_gen = DocumentationGenerator()
    dashboard = DashboardUI()

    # ─── State ────────────────────────────────────────────────────────
    show_dashboard = config.ENABLE_DASHBOARD
    show_landmarks = True
    logged_in = False

    # Default driver login
    login_info = profile.login(config.DEFAULT_DRIVER_NAME)
    logged_in = True
    print(f"\n  Logged in as: {login_info['name']}")
    print(f"  Session ID: {login_info['session_id']}")
    print(f"  Calibrated: {login_info['has_calibration']}")
    print(f"  Fatigue pattern: {login_info['fatigue_pattern']}")

    # Apply calibrated thresholds
    eyes.set_threshold(profile.ear_threshold)
    yawn.set_threshold(profile.mar_threshold)
    head.set_baseline(profile.pitch_baseline, profile.yaw_baseline)

    # ─── Open camera ──────────────────────────────────────────────────
    print("\n  Opening camera...")
    if not video.open_camera():
        print("  ERROR: Could not open camera!")
        print("  Attempting reconnect...")
        if not video.reconnect_camera():
            print("  Failed to connect to camera. Exiting.")
            return

    print("  Camera ready!")
    print("\n  Controls: ESC=Quit | C=Calibrate | D=Dashboard | A=Acknowledge | X=Dismiss")
    print("  R=Export Report | G=Generate Docs | 1-9=Driver profiles")
    print("-" * 60)

    # Start session logging
    logger.start_session(profile.session_id, profile.current_driver_id)

    # Tracking for session end
    max_score = 0
    total_alerts = 0

    # ─── Main loop ────────────────────────────────────────────────────
    try:
        while True:
            # Read frame
            ret, frame, fps = video.read_frame()

            if not ret:
                if not video.is_camera_healthy:
                    print("  Camera lost. Attempting reconnect...")
                    if video.reconnect_camera():
                        continue
                    else:
                        print("  Reconnect failed. Exiting.")
                        break
                continue

            h, w, _ = frame.shape

            # Skip frame if FPS is too low (to catch up)
            if video.should_skip_frame():
                cv2.waitKey(1)
                continue

            # ─── Process face detection ───────────────────────────────
            face, processed_frame = video.process_frame(frame)

            # ─── Module 02: Eye Detection ─────────────────────────────
            eye_data = eyes.update(
                face.landmark if face else None, w, h
            )

            # ─── Module 03: Yawn Detection ────────────────────────────
            yawn_data = yawn.update(
                face.landmark if face else None, w, h
            )

            # ─── Module 04: Head Pose ─────────────────────────────────
            head_data = head.update(
                face.landmark if face else None, w, h
            )

            # ─── Module 10: Occlusion Handling ────────────────────────
            occlusion_data = occlusion.update(
                face, w, h,
                eye_data=eye_data,
                yawn_data=yawn_data,
                video_processor=video,
            )

            # ─── Module 06: Environment ───────────────────────────────
            has_glare = video.detect_glare(frame, face, w, h) if face else False
            environment.update_lighting(video.avg_brightness, has_glare)
            env_data = environment.get_status()

            # ─── Module 05: Calibration ───────────────────────────────
            if profile.is_calibrating:
                profile.add_calibration_frame(
                    eye_data["ear"], yawn_data["mar"],
                    head_data["pitch"], head_data["yaw"]
                )
                cal_done, cal_progress = profile.check_calibration_complete()

                # Show calibration progress overlay
                bar_w = int(cal_progress * w)
                cv2.rectangle(frame, (0, h - 30), (bar_w, h), (0, 200, 0), -1)
                cv2.putText(frame, f"Calibrating... {cal_progress:.0%}",
                             (20, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                             0.6, (255, 255, 255), 2)

                if cal_done:
                    print(f"  Calibration complete!")
                    print(f"  EAR threshold: {profile.ear_threshold:.3f}")
                    print(f"  MAR threshold: {profile.mar_threshold:.3f}")
                    eyes.set_threshold(profile.ear_threshold)
                    yawn.set_threshold(profile.mar_threshold)
                    head.set_baseline(profile.pitch_baseline, profile.yaw_baseline)

            # ─── Module 07: Temporal Model ────────────────────────────
            temporal.add_features(
                eye_data["ear"], yawn_data["mar"],
                head_data["pitch"], head_data["yaw"],
                eye_data["blink_rate_bpm"],
                yawn_data["yawn_frequency_5min"],
            )
            fatigue_prob, fatigue_conf = temporal.predict()
            temporal_data = temporal.get_status()

            # ─── Module 08: Scoring Engine ────────────────────────────
            # Adjust thresholds based on environment and driver history
            env_adjust = environment.get_threshold_adjustment()
            driver_adjust = profile.get_adaptive_sensitivity()
            scorer.adjust_thresholds(env_adjust, driver_adjust)

            # Build signals and compute score
            signals = scorer.build_signal_dict(
                eye_data, yawn_data, head_data, env_data, fps
            )

            # Apply occlusion confidence multipliers
            conf_mult = occlusion.get_confidence_multipliers()
            for sig_name in signals:
                if sig_name in ("ear", "perclos", "blink_rate"):
                    signals[sig_name]["confidence"] *= conf_mult.get("eyes", 1.0)
                elif sig_name == "yawn_freq":
                    signals[sig_name]["confidence"] *= conf_mult.get("mouth", 1.0)
                elif sig_name == "head_pose":
                    signals[sig_name]["confidence"] *= conf_mult.get("head_pose", 1.0)

            score_data = scorer.compute_score(signals)
            global latest_data
            latest_data = {
                "ear": float(eye_data["ear"]),
                "mar": float(yawn_data["mar"]),
                "score": float(score_data["score"]),                            
                "fatigue": float(fatigue_prob * 100)
            }
            max_score = max(max_score, score_data["score"])

            # ─── Module 09: Alert System ──────────────────────────────
            alert_data = alerts.check_and_alert(
                score_data, eye_data, yawn_data, head_data,
                env_data, profile
            )
            if alert_data["triggered"]:
                total_alerts += 1
                profile.log_event(
                    "alert", alert_data["level"],
                    "; ".join(alert_data["reasons"]),
                    score_data["score"]
                )
                logger.log_alert({
                    "level": alert_data["level"],
                    "reasons": alert_data["reasons"],
                    "score": score_data["score"],
                })

            # ─── Draw on video frame ──────────────────────────────────
            if show_landmarks and face:
                frame = video.draw_landmarks(frame, face)
                frame = head.draw_pose_axes(frame, face.landmark, w, h)

            # Draw EAR/MAR values on video
            ear_color = (0, 255, 0) if not eye_data["eye_closed"] else (0, 0, 255)
            cv2.putText(frame, f"EAR: {eye_data['ear']:.2f}", (10, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_color, 2)
            cv2.putText(frame, f"MAR: {yawn_data['mar']:.2f}", (10, 55),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Score and level
            score_color = alerts.get_alert_color(score_data["level"])
            cv2.putText(frame, f"Score: {score_data['score']:.0f}/100",
                         (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, score_color, 2)

            # FPS
            cv2.putText(frame, f"FPS: {fps:.0f}", (w - 100, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Driver name and trip duration
            trip = env_data["trip_display"]
            cv2.putText(frame, f"{profile.current_driver_name} | {trip}",
                         (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                         0.45, (200, 200, 200), 1)

            # Alert text on video
            if alert_data["triggered"]:
                text = alert_data["display_text"]
                color = alerts.get_alert_color(alert_data["level"])
                cv2.putText(frame, text, (10, 120),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Show top reason
                reasons = alert_data.get("reasons", [])
                if reasons:
                    cv2.putText(frame, reasons[0], (10, 145),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # ─── Module 12: Frame Logging ─────────────────────────────
            logger.log_frame({
                "ear": eye_data["ear"],
                "mar": yawn_data["mar"],
                "pitch": head_data["pitch"],
                "yaw": head_data["yaw"],
                "score": score_data["score"],
                "level": score_data["level"],
                "perclos": eye_data["perclos"],
                "blink_rate": eye_data["blink_rate_bpm"],
                "fps": fps,
            })

            # ─── Module 11: Dashboard ─────────────────────────────────
            if show_dashboard:
                dashboard.update_data(
                    ear=eye_data["ear"],
                    mar=yawn_data["mar"],
                    pitch=head_data["pitch"],
                    yaw=head_data["yaw"],
                    score=score_data["score"],
                    fps=fps,
                )
                display = dashboard.render_dashboard(
                    frame, eye_data, yawn_data, head_data,
                    score_data, alert_data, env_data,
                    occlusion_data, temporal_data,
                    {"name": profile.current_driver_name},
                    fps,
                )
                cv2.imshow("Driver Drowsiness Detection System", display)
            else:
                cv2.imshow("Driver Drowsiness Detection System", frame)

            # ─── Key handling ─────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                global camera_running
                camera_running = False  # ESC
                break

            elif key == ord('c') or key == ord('C'):
                print("  Starting 30-second calibration...")
                print("  Please look straight ahead with eyes open.")
                profile.start_calibration()

            elif key == ord('d') or key == ord('D'):
                show_dashboard = not show_dashboard
                if not show_dashboard:
                    cv2.destroyAllWindows()
                print(f"  Dashboard: {'ON' if show_dashboard else 'OFF'}")

            elif key == ord('l') or key == ord('L'):
                show_landmarks = not show_landmarks
                print(f"  Landmarks: {'ON' if show_landmarks else 'OFF'}")

            elif key == ord('a') or key == ord('A'):
                alerts.acknowledge_alert()
                print("  Alert acknowledged")

            elif key == ord('x') or key == ord('X'):
                alerts.dismiss_alert(profile)
                dashboard.log_feedback(is_true_alarm=False)
                logger.log_feedback(is_true_alarm=False)
                print("  Alert dismissed (false alarm logged)")

            elif key == ord('f') or key == ord('F'):
                dashboard.log_feedback(is_true_alarm=True)
                logger.log_feedback(is_true_alarm=True)
                print("  Feedback: true alarm logged")

            elif key == ord('r') or key == ord('R'):
                csv_dir = logger.export_csv()
                json_path = logger.export_json()
                report_path = logger.export_pdf_report()
                print(f"  Reports exported to: {csv_dir}")
                print(f"  Summary: {json_path}")

            elif key == ord('g') or key == ord('G'):
                paths = doc_gen.generate_all_docs()
                print(f"  Documentation generated:")
                for name, path in paths.items():
                    print(f"    {name}: {path}")

            elif ord('1') <= key <= ord('9'):
                driver_num = key - ord('0')
                driver_name = f"Driver_{driver_num}"
                login_info = profile.login(driver_name)
                eyes.set_threshold(profile.ear_threshold)
                yawn.set_threshold(profile.mar_threshold)
                head.set_baseline(profile.pitch_baseline, profile.yaw_baseline)
                print(f"  Switched to: {driver_name}")
                print(f"  Calibrated: {login_info['has_calibration']}")

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")

    finally:
        # ─── Cleanup ──────────────────────────────────────────────────
        print("\n  Saving session data...")
        profile.end_session(
            max_score=max_score,
            total_alerts=total_alerts,
            total_yawns=yawn.yawn_count,
            total_microsleeps=eye_data.get("microsleep_count", 0) if 'eye_data' in dir() else 0,
            avg_perclos=eyes.perclos_value,
        )

        # Auto-export session reports
        try:
            logger.export_csv()
            logger.export_json()
            logger.export_pdf_report()
            print("  Reports exported to ./reports/")
        except Exception as e:
            print(f"  Report export error: {e}")

        profile.close()
        video.release()
        cv2.destroyAllWindows()
        print("  Session saved. Goodbye!")


if __name__ == "__main__":
    main()
