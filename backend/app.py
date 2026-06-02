"""
Flask API Server for Smart Hostel Management System - Version 6

Features:
- Live camera recognition
- Multiple images per student
- Delete student
- CSV bulk import
- Registered students with image count
"""

from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import cv2
import csv
import shutil
import uuid
import numpy as np
import face_recognition
import threading
import time

from database import Database
from anomaly_detector import AnomalyDetector


# ─────────────────────────────────────────────────────────────────────────────
# Flask Setup
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, '../frontend'))

app = Flask(
    __name__,
    template_folder=FRONTEND_DIR,
    static_folder=FRONTEND_DIR
)

CORS(app)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'images')
BULK_UPLOAD_FOLDER = os.path.join(BASE_DIR, 'bulk_upload')

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
ALLOWED_CSV_EXTENSIONS = {'csv'}

MANUAL_CAMERA_INDEX = 0

# This is a similarity threshold, not a real probability.
# 45 is slightly more practical for demo conditions.
CONFIDENCE_THRESHOLD = 45.0

# face_recognition tolerance.
# Higher = more relaxed, lower = stricter.
FACE_TOLERANCE = 0.60

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(BULK_UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['BULK_UPLOAD_FOLDER'] = BULK_UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024

Database.init_db()


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return (
        filename and
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def allowed_csv(filename):
    return (
        filename and
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_CSV_EXTENSIONS
    )


def make_unique_image_filename(student_name, original_filename):
    """
    Create safe unique image filename.
    """
    safe_name = secure_filename(student_name.strip().replace(" ", "_"))

    if not safe_name:
        safe_name = "student"

    ext = original_filename.rsplit('.', 1)[1].lower()
    unique_id = uuid.uuid4().hex[:8]

    return f"{safe_name}_{int(time.time() * 1000)}_{unique_id}.{ext}"


def save_uploaded_image(file, student_name):
    """
    Save uploaded image into backend/images.
    """
    filename = make_unique_image_filename(student_name, file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    return filepath


def copy_bulk_image_to_uploads(source_path, student_name):
    """
    Copy an image from backend/bulk_upload to backend/images.
    """
    original_filename = os.path.basename(source_path)
    filename = make_unique_image_filename(student_name, original_filename)

    destination_path = os.path.join(UPLOAD_FOLDER, filename)
    shutil.copy2(source_path, destination_path)

    return destination_path


# ─────────────────────────────────────────────────────────────────────────────
# Camera Recognition
# ─────────────────────────────────────────────────────────────────────────────

class CameraRecognizer:
    def __init__(self):
        self.cap = None

        # Multiple encodings allowed.
        # known_names may contain same student name multiple times.
        self.known_encodings = []
        self.known_names = []
        self.known_user_ids = []
        self.known_image_paths = []

        self.last_logged = {}
        self.lock = threading.Lock()
        self.running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()

        self.load_users()

    def load_users(self):
        """
        Load all registered student images as face encodings.
        """
        with self.lock:
            self.known_encodings = []
            self.known_names = []
            self.known_user_ids = []
            self.known_image_paths = []

            rows = Database.get_all_user_images_for_recognition()

            loaded_count = 0
            skipped_count = 0

            for user_id, name, path in rows:
                try:
                    if not os.path.exists(path):
                        print(f"[WARN] Image not found for {name}: {path}")
                        skipped_count += 1
                        continue

                    img = face_recognition.load_image_file(path)
                    encodings = face_recognition.face_encodings(img)

                    if not encodings:
                        print(f"[WARN] No face found in image for {name}: {path}")
                        skipped_count += 1
                        continue

                    # Use first face found
                    self.known_encodings.append(encodings[0])
                    self.known_names.append(name)
                    self.known_user_ids.append(user_id)
                    self.known_image_paths.append(path)

                    loaded_count += 1

                except Exception as e:
                    print(f"[WARN] Error loading {name} image {path}: {e}")
                    skipped_count += 1

            print(
                f"[INFO] Face encodings loaded: {loaded_count}, skipped: {skipped_count}"
            )

    def start_camera(self):
        if not self.running:
            self.running = True
            thread = threading.Thread(target=self._camera_loop, daemon=True)
            thread.start()
            print("[INFO] Camera thread started")

    def stop_camera(self):
        self.running = False

        if self.cap:
            self.cap.release()

    def _camera_loop(self):
        self.cap = cv2.VideoCapture(MANUAL_CAMERA_INDEX)

        if not self.cap.isOpened():
            print("[ERROR] Camera not found")
            self.running = False
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        print("[INFO] Camera stream started")

        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                time.sleep(0.05)
                continue

            try:
                frame = self._process_frame(frame)

                with self.frame_lock:
                    self.current_frame = frame.copy()

            except Exception as e:
                print(f"[ERROR] Frame processing: {e}")

            time.sleep(0.03)

        self.cap.release()
        print("[INFO] Camera stream stopped")

    def _safe_to_rgb(self, frame):
        if frame is None:
            raise ValueError("Frame is None")

        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        if frame.ndim == 3:
            if frame.shape[2] == 3:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif frame.shape[2] == 4:
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        raise ValueError("Unsupported frame format")

    def _process_frame(self, frame):
        try:
            rgb_frame = self._safe_to_rgb(frame)

            face_locations = face_recognition.face_locations(
                rgb_frame,
                model='hog'
            )

            face_encodings = face_recognition.face_encodings(
                rgb_frame,
                face_locations
            )

            for face_encoding, (top, right, bottom, left) in zip(
                face_encodings,
                face_locations
            ):
                with self.lock:
                    if not self.known_encodings:
                        name = "Unknown"
                        confidence = 0.0
                    else:
                        distances = face_recognition.face_distance(
                            self.known_encodings,
                            face_encoding
                        )

                        best_index = int(np.argmin(distances))
                        best_distance = float(distances[best_index])

                        confidence = float((1 - best_distance) * 100)

                        match_ok = (
                            best_distance <= FACE_TOLERANCE and
                            confidence >= CONFIDENCE_THRESHOLD
                        )

                        if match_ok:
                            name = self.known_names[best_index]
                        else:
                            name = "Unknown"

                anomaly_info = AnomalyDetector.detect(name, confidence)

                current_time = time.time()

                # Log same visible name every 5 seconds.
                if name not in self.last_logged or current_time - self.last_logged[name] > 5:
                    status = "Authorized" if name != "Unknown" else "Unauthorized"

                    Database.insert_log(
                        name=name,
                        status=status,
                        confidence=confidence,
                        anomaly=1 if anomaly_info["is_anomaly"] else 0
                    )

                    self.last_logged[name] = current_time

                    if anomaly_info["is_anomaly"]:
                        print(
                            f"🚨 ANOMALY: {anomaly_info['type']} - "
                            f"{name} ({confidence:.2f}%)"
                        )

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

                cv2.rectangle(
                    frame,
                    (left, top),
                    (right, bottom),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"{name} ({confidence:.1f}%)",
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

        except Exception as e:
            print(f"[ERROR] Processing: {e}")

        return frame

    def get_frame(self):
        with self.frame_lock:
            if self.current_frame is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)

                cv2.putText(
                    blank,
                    "Camera Initializing...",
                    (150, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

                return blank

            return self.current_frame.copy()


camera = CameraRecognizer()


# ─────────────────────────────────────────────────────────────────────────────
# Response Headers
# ─────────────────────────────────────────────────────────────────────────────

@app.after_request
def add_no_cache_headers(response):
    if request.path.startswith(('/logs', '/users', '/stats')):
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'

    return response


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('alerts.html')


@app.route('/video_feed')
def video_feed():
    def gen_frames():
        boundary = b'--frame\r\n'
        while True:
            try:
                frame = camera.get_frame()

                if frame is None:
                    time.sleep(0.05)
                    continue

                ret, buffer = cv2.imencode(
                    '.jpg',
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 75]
                )

                if not ret:
                    time.sleep(0.05)
                    continue

                yield (
                    boundary +
                    b'Content-Type: image/jpeg\r\n'
                    b'Content-Length: ' + str(len(buffer)).encode() + b'\r\n\r\n' +
                    buffer.tobytes() +
                    b'\r\n'
                )

                time.sleep(0.04)  # ~25 FPS, gives breathing room

            except (GeneratorExit, BrokenPipeError, ConnectionResetError):
                # Client disconnected — exit gracefully WITHOUT crashing
                print("[INFO] MJPEG client disconnected gracefully")
                break
            except Exception as e:
                print(f"[WARN] MJPEG generator error (continuing): {e}")
                time.sleep(0.1)
                continue

    response = Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['X-Accel-Buffering'] = 'no'  # Disable proxy buffering
    response.headers['Connection'] = 'close'  # Don't reuse this connection
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Logs
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/logs', methods=['GET'])
def get_logs():
    try:
        name = request.args.get('name', '').strip()
        timestamp = request.args.get('timestamp', '').strip()
        status = request.args.get('status', '').strip()
        anomaly = request.args.get('anomaly', '').strip()
        limit = request.args.get('limit', '100').strip()

        try:
            limit = int(limit)
        except ValueError:
            limit = 100

        anomaly_value = None

        if anomaly in ('0', '1'):
            anomaly_value = int(anomaly)

        logs = Database.get_logs(
            limit=limit,
            name=name,
            timestamp=timestamp,
            status=status,
            anomaly=anomaly_value
        )

        return jsonify([
            {
                "id": l[0],
                "name": l[1],
                "timestamp": l[2],
                "status": l[3],
                "confidence": round(float(l[4]), 2),
                "anomaly": int(l[5])
            }
            for l in logs
        ])

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Users
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/users', methods=['GET'])
def get_users():
    try:
        users = Database.get_all_users()

        return jsonify([
            {
                "id": u[0],
                "name": u[1],
                "image_count": int(u[3]) if u[3] is not None else 0,
                "created_at": u[4],
                "primary_image_url": f"/user_image/{u[0]}"
            }
            for u in users
        ])

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/user_image/<int:user_id>', methods=['GET'])
def user_image(user_id):
    """
    Serve primary image for a student.
    """
    try:
        images = Database.get_user_images(user_id)

        if not images:
            return jsonify({"error": "No image found"}), 404

        image_path = images[0][2]

        if not os.path.exists(image_path):
            return jsonify({"error": "Image file missing"}), 404

        directory = os.path.dirname(image_path)
        filename = os.path.basename(image_path)

        return send_from_directory(directory, filename)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/add_user', methods=['POST'])
def add_user():
    """
    Add a student with one or multiple images.

    If student already exists, uploaded images are added to existing student.
    """
    try:
        if 'name' not in request.form:
            return jsonify({"error": "Missing name"}), 400

        name = request.form['name'].strip()

        if not name:
            return jsonify({"error": "Invalid name"}), 400

        files = request.files.getlist('images')

        # backward compatibility if frontend sends "image"
        if not files:
            files = request.files.getlist('image')

        if not files or files[0].filename == '':
            return jsonify({"error": "Please select at least one image"}), 400

        saved_paths = []

        for file in files:
            if not file or file.filename == '':
                continue

            if not allowed_file(file.filename):
                return jsonify({
                    "error": f"Invalid file type: {file.filename}"
                }), 400

            filepath = save_uploaded_image(file, name)
            saved_paths.append(filepath)

        if not saved_paths:
            return jsonify({"error": "No valid images uploaded"}), 400

        result = Database.add_user_with_images(name, saved_paths)

        camera.load_users()

        message = (
            "Student added successfully"
            if result["created"]
            else "Images added to existing student"
        )

        return jsonify({
            "message": message,
            "name": name,
            "user_id": result["user_id"],
            "images_added": result["images_added"],
            "created": result["created"]
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/delete_user/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    """
    Delete user from database only.
    Physical image files are kept.
    """
    try:
        deleted = Database.delete_user(user_id)

        if not deleted:
            return jsonify({"error": "Student not found"}), 404

        camera.load_users()

        return jsonify({
            "message": "Student deleted from database successfully",
            "user_id": user_id
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# CSV Bulk Import
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/import_csv', methods=['POST'])
def import_csv():
    """
    Import students from CSV.

    CSV format:
    name,image

    Images must already exist inside:
    backend/bulk_upload/
    """
    try:
        if 'csv_file' not in request.files:
            return jsonify({"error": "Missing CSV file"}), 400

        csv_file = request.files['csv_file']

        if csv_file.filename == '':
            return jsonify({"error": "No CSV file selected"}), 400

        if not allowed_csv(csv_file.filename):
            return jsonify({"error": "Only .csv file is allowed"}), 400

        decoded = csv_file.stream.read().decode("utf-8-sig").splitlines()
        reader = csv.DictReader(decoded)

        required_columns = {'name', 'image'}

        if not reader.fieldnames or not required_columns.issubset(set(reader.fieldnames)):
            return jsonify({
                "error": "CSV must contain columns: name,image"
            }), 400

        imported_students = {}
        skipped = []
        total_images_added = 0

        for row_number, row in enumerate(reader, start=2):
            name = (row.get('name') or '').strip()
            image_name = (row.get('image') or '').strip()

            if not name or not image_name:
                skipped.append({
                    "row": row_number,
                    "reason": "Missing name or image"
                })
                continue

            image_basename = os.path.basename(image_name)
            source_path = os.path.join(BULK_UPLOAD_FOLDER, image_basename)

            if not os.path.exists(source_path):
                skipped.append({
                    "row": row_number,
                    "name": name,
                    "image": image_name,
                    "reason": "Image not found in backend/bulk_upload"
                })
                continue

            if not allowed_file(image_basename):
                skipped.append({
                    "row": row_number,
                    "name": name,
                    "image": image_name,
                    "reason": "Invalid image type"
                })
                continue

            copied_path = copy_bulk_image_to_uploads(source_path, name)

            result = Database.add_user_with_images(name, [copied_path])

            if name not in imported_students:
                imported_students[name] = {
                    "user_id": result["user_id"],
                    "images_added": 0
                }

            imported_students[name]["images_added"] += 1
            total_images_added += 1

        camera.load_users()

        return jsonify({
            "message": "CSV import completed",
            "students_imported_or_updated": len(imported_students),
            "images_added": total_images_added,
            "students": imported_students,
            "skipped": skipped
        }), 200

    except UnicodeDecodeError:
        return jsonify({
            "error": "CSV encoding error. Please save CSV as UTF-8."
        }), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        logs = Database.get_logs(limit=100000)
        users = Database.get_all_users()

        total_images = sum(int(u[3]) for u in users)

        return jsonify({
            "total_logs": len(logs),
            "total_users": len(users),
            "total_images": total_images,
            "unknown_entries": len([l for l in logs if l[1] == "Unknown"]),
            "anomalies": len([l for l in logs if int(l[5]) == 1])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/refresh_logs', methods=['POST'])
def refresh_logs():
    try:
        logs = Database.get_logs()

        return jsonify({
            "message": "Logs refreshed",
            "count": len(logs)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Camera Autostart
# ─────────────────────────────────────────────────────────────────────────────

@app.before_request
def before_request():
    # Only start camera on actual page/feed requests, not every API call
    if request.endpoint == 'video_feed' and not camera.running:
        camera.start_camera()
    elif request.endpoint == 'index' and not camera.running:
        camera.start_camera()


if __name__ == '__main__':
    print("\n[INFO] Starting Smart Hostel Management System - Version 6")
    print("[INFO] Access frontend at http://localhost:5000")
    print(f"[INFO] Upload folder: {UPLOAD_FOLDER}")
    print(f"[INFO] Bulk upload folder: {BULK_UPLOAD_FOLDER}\n")

    app.run(
        debug=False,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )