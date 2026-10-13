"""
Ai Based Synthetic Media Detection System - Deepfake Detection & Verification System.
Integrates all 12 Architectural Modules:
1. User Authentication
2. Image & Media Upload
3. Image Preprocessing & Face Isolation
4. Deepfake Detection Model (PyTorch ResNet-18)
5. Confidence & Anomaly Scoring
6. Explainable AI (Grad-CAM Heatmaps)
7. Result Dashboard
8. Detection History Audit Trail
9. Downloadable Forensic PDF Dossiers
10. Database Layer (MySQL with SQLite Auto-Fallback)
11. Administration Suite & Triage
12. Security & Parameter Validation
"""

import os
import uuid
import shutil
from flask import (
    Flask, request, render_template, send_from_directory, redirect,
    url_for, session, flash, abort
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

import db
from core.auth import login_required, admin_required, validate_registration
from core.detector import DeepfakeDetector
from core.report_gen import generate_pdf_report

# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "aegis_forensics_secure_vault_key_2026")
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max upload cap

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
REPORTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Initialize database
db.init_db()

# Load Deepfake Detection Engine (ResNet-18 + Grad-CAM)
detector = DeepfakeDetector()


def allowed_file(filename, media_type="image"):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if media_type == "video":
        return ext in ALLOWED_VIDEO_EXTENSIONS
    return ext in ALLOWED_IMAGE_EXTENSIONS


# -------------------------------------------------------------------
# FILE SERVING
# -------------------------------------------------------------------
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/reports/<path:filename>')
def serve_report(filename):
    return send_from_directory(REPORTS_FOLDER, filename)


# -------------------------------------------------------------------
# AUTHENTICATION MODULE (Module 1 & 12)
# -------------------------------------------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        user = db.get_user_by_username(username)
        if user and check_password_hash(user['password_hash'], password):
            session.clear()
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['full_name'] = user['full_name']
            session['role'] = user['role']
            session['admin'] = (user['role'] == 'admin')

            flash(f"Welcome back, {user['full_name'] or user['username']}.", "success")
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            if user['role'] == 'admin':
                return redirect(url_for('admin_panel'))
            return redirect(url_for('home'))

        flash("Invalid credentials. Please verify your username and password.", "danger")

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        is_valid, err_msg = validate_registration(username, email, password, confirm_password)
        if not is_valid:
            flash(err_msg, "danger")
            return render_template('register.html')

        pw_hash = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)
        user_id = db.create_user(
            username=username,
            email=email,
            password_hash=pw_hash,
            full_name=full_name,
            role='user'
        )

        session.clear()
        session['user_id'] = user_id
        session['username'] = username
        session['full_name'] = full_name
        session['role'] = 'user'
        session['admin'] = False

        flash("Account created successfully. Analyst clearance granted.", "success")
        return redirect(url_for('home'))

    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    flash("Session terminated safely.", "info")
    return redirect(url_for('home'))


@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    return redirect(url_for('login'))


# -------------------------------------------------------------------
# SCAN & ANALYSIS MODULES (Modules 2, 3, 4, 5, 6, 7)
# -------------------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/scan', methods=['POST'])
def scan_media():
    try:
        if 'file' not in request.files:
            flash("No media file submitted for examination.", "warning")
            return redirect(url_for('home'))

        file = request.files['file']
        if file.filename == '':
            flash("No file selected.", "warning")
            return redirect(url_for('home'))

        media_type = request.form.get('media_type', 'image')
        if not allowed_file(file.filename, media_type):
            flash(f"Unsupported file format for {media_type} analysis.", "danger")
            return redirect(url_for('home'))

        # Secure filename generation
        safe_orig_name = secure_filename(file.filename)
        unique_prefix = uuid.uuid4().hex[:12]
        saved_filename = f"{unique_prefix}_{safe_orig_name}"
        save_path = os.path.join(UPLOAD_FOLDER, saved_filename)
        file.save(save_path)

        # Run Deepfake Forensic Engine
        if media_type == 'video':
            res = detector.analyze_video(save_path, output_dir=UPLOAD_FOLDER)
        else:
            res = detector.analyze_image(save_path, output_dir=UPLOAD_FOLDER)

        # Persist Scan in Database (Module 8 & 10)
        user_id = session.get('user_id')
        scan_id = db.create_scan(
            user_id=user_id,
            filename=saved_filename,
            original_name=safe_orig_name,
            media_type=media_type,
            verdict=res['verdict'],
            confidence=res['confidence'],
            fake_prob=res['fake_prob'],
            real_prob=res['real_prob'],
            face_detected=res.get('face_detected', True),
            face_crop_path=res.get('face_crop_filename'),
            heatmap_path=res.get('heatmap_filename'),
            overlay_path=res.get('overlay_filename'),
            details_json={
                'suspicious_regions': res.get('suspicious_regions', []),
                'metrics': res.get('metrics', {})
            }
        )

        return redirect(url_for('view_result', scan_id=scan_id))

    except Exception as e:
        flash(f"Forensic scan error: {str(e)}", "danger")
        return redirect(url_for('home'))


@app.route('/scan_sample', methods=['POST'])
def scan_sample():
    """Execute quick benchmark test with sample images."""
    try:
        sample_name = request.form.get('sample_name', 'fake.png')
        sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_images', sample_name)
        if not os.path.exists(sample_path):
            flash(f"Sample file {sample_name} not found.", "warning")
            return redirect(url_for('home'))

        unique_prefix = uuid.uuid4().hex[:12]
        saved_filename = f"sample_{unique_prefix}_{sample_name}"
        save_path = os.path.join(UPLOAD_FOLDER, saved_filename)
        shutil.copyfile(sample_path, save_path)

        res = detector.analyze_image(save_path, output_dir=UPLOAD_FOLDER)

        user_id = session.get('user_id')
        scan_id = db.create_scan(
            user_id=user_id,
            filename=saved_filename,
            original_name=sample_name,
            media_type='image',
            verdict=res['verdict'],
            confidence=res['confidence'],
            fake_prob=res['fake_prob'],
            real_prob=res['real_prob'],
            face_detected=res.get('face_detected', True),
            face_crop_path=res.get('face_crop_filename'),
            heatmap_path=res.get('heatmap_filename'),
            overlay_path=res.get('overlay_filename'),
            details_json={
                'suspicious_regions': res.get('suspicious_regions', []),
                'metrics': res.get('metrics', {})
            }
        )
        return redirect(url_for('view_result', scan_id=scan_id))
    except Exception as e:
        flash(f"Sample scan error: {str(e)}", "danger")
        return redirect(url_for('home'))


@app.route('/result/<int:scan_id>')
def view_result(scan_id):
    scan = db.get_scan_by_id(scan_id)
    if not scan:
        flash("Forensic dossier record not found.", "warning")
        return redirect(url_for('home'))
    return render_template('result.html', scan=scan)


# -------------------------------------------------------------------
# REPORT GENERATION MODULE (Module 9)
# -------------------------------------------------------------------
@app.route('/download_report/<int:scan_id>')
def download_report(scan_id):
    try:
        pdf_path = generate_pdf_report(scan_id, uploads_dir=UPLOAD_FOLDER, reports_dir=REPORTS_FOLDER)
        filename = os.path.basename(pdf_path)
        return send_from_directory(REPORTS_FOLDER, filename, as_attachment=True)
    except Exception as e:
        flash(f"Report generation error: {str(e)}", "danger")
        return redirect(url_for('view_result', scan_id=scan_id))


# -------------------------------------------------------------------
# DETECTION HISTORY MODULE (Module 8)
# -------------------------------------------------------------------
@app.route('/history')
def history():
    user_id = session.get('user_id')
    if user_id:
        scans = db.get_user_scans(user_id)
    else:
        # Fallback: display latest public/session scans
        scans = db.get_all_scans(limit=15)
    return render_template('history.html', scans=scans)


@app.route('/scan/delete/<int:scan_id>', methods=['POST'])
def delete_user_scan(scan_id):
    user_id = session.get('user_id')
    if session.get('role') == 'admin':
        db.delete_scan(scan_id)
    elif user_id:
        db.delete_scan(scan_id, user_id=user_id)
    flash("Scan record removed from history.", "info")
    return redirect(url_for('history'))


# -------------------------------------------------------------------
# [FEATURE COMING SOON] INCIDENT COMPLAINTS & LAW ENFORCEMENT ESCALATION
# NOTE: The "Report Crime" / Cybercrime Escalation feature is reserved
# for a future platform update. All UI entry points (navigation bar,
# result actions, and admin triage tabs) have been commented out.
# The backend handlers and business logic below are preserved intact
# for immediate activation when the feature is rolled out.
# -------------------------------------------------------------------
@app.route('/complaint')
def complaint():
    """
    [FUTURE FEATURE - COMING SOON]
    Cybercrime Incident Reporting Portal.
    Renders 'complaint.html' which currently displays a 'Feature Scheduled
    for Future Release' notice while keeping the entire submission form commented.
    """
    scan_id = request.args.get('scan_id')
    linked_scan = None
    if scan_id:
        try:
            linked_scan = db.get_scan_by_id(int(scan_id))
        except Exception:
            pass
    return render_template('complaint.html', linked_scan=linked_scan)


@app.route('/submit_complaint', methods=['POST'])
def submit_complaint():
    """
    [FUTURE FEATURE - COMING SOON]
    Process and record citizen / user deepfake fraud complaints.
    Stores complainant identity, description, and attached evidence.
    Will be fully linked with law enforcement escalation APIs in an upcoming release.
    """
    try:
        name = request.form.get('name', '').strip()
        age = request.form.get('age', '').strip()
        gender = request.form.get('gender', '')
        mobile = request.form.get('mobile', '').strip()
        desc = request.form.get('desc', '').strip()
        scan_id = request.form.get('scan_id')
        user_id = session.get('user_id')

        img_name = None
        file = request.files.get('file')
        if file and file.filename != '':
            img_name = f"proof_{uuid.uuid4().hex[:12]}_{secure_filename(file.filename)}"
            file.save(os.path.join(UPLOAD_FOLDER, img_name))
        elif scan_id:
            # Attach the scan's original file as primary proof
            scan = db.get_scan_by_id(int(scan_id))
            if scan:
                img_name = scan['filename']

        db.create_complaint(
            name=name,
            age=age,
            gender=gender,
            mobile=mobile,
            description=desc,
            image=img_name,
            user_id=user_id,
            scan_id=scan_id if scan_id else None
        )

        flash("Incident report submitted successfully. Law enforcement dossier created.", "success")
        return redirect(url_for('complaint'))

    except Exception as e:
        flash(f"Failed to submit complaint: {str(e)}", "danger")
        return redirect(url_for('complaint'))


# -------------------------------------------------------------------
# ADMIN DASHBOARD MODULE (Module 11)
# -------------------------------------------------------------------
@app.route('/panel')
@app.route('/admin')
@admin_required
def admin_panel():
    stats = db.get_admin_stats()
    complaints = db.get_all_complaints()
    scans = db.get_all_scans(limit=50)
    users = db.get_all_users()
    return render_template('admin_panel.html', stats=stats, complaints=complaints, scans=scans, users=users)


# -------------------------------------------------------------------
# [FUTURE FEATURE - COMING SOON] ADMIN COMPLAINTS RESOLUTION / TRIAGE
# Preserved for future release when the complaints workflow is enabled.
# -------------------------------------------------------------------
@app.route('/admin/resolve/<int:id>')
@app.route('/resolve/<int:id>')
@admin_required
def resolve_complaint(id):
    """[FUTURE FEATURE] Update status of incident complaint dossier to Resolved."""
    db.update_complaint_status(id, 'Resolved')
    flash(f"Complaint #{id} marked as Resolved.", "success")
    return redirect(url_for('admin_panel'))


@app.route('/admin/delete/<int:id>')
@app.route('/delete/<int:id>')
@admin_required
def delete_complaint_admin(id):
    """[FUTURE FEATURE] Permanently remove an incident complaint record."""
    db.delete_complaint(id)
    flash(f"Complaint #{id} removed.", "info")
    return redirect(url_for('admin_panel'))


@app.route('/admin/user/delete/<int:id>', methods=['POST'])
@admin_required
def delete_user_admin(id):
    db.delete_user(id)
    flash(f"User #{id} account removed.", "info")
    return redirect(url_for('admin_panel'))


# -------------------------------------------------------------------
# ERROR HANDLERS
# -------------------------------------------------------------------
@app.errorhandler(413)
def request_entity_too_large(error):
    flash("The uploaded file exceeds the 50MB payload limit.", "danger")
    return redirect(url_for('home'))


@app.errorhandler(404)
def page_not_found(error):
    flash("Requested forensic resource was not found.", "warning")
    return redirect(url_for('home'))


# -------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)