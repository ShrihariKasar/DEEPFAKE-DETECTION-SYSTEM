"""
Database Module for Deepfake Detection & Forensic System.
Supports MySQL with automatic SQLite fallback for high reliability.
Handles Users, Scans (History), Complaints, and Administrative records.
"""

import os
import sqlite3
import json
from datetime import datetime

# Attempt MySQL connector
try:
    import mysql.connector
    from mysql.connector import pooling
    HAS_MYSQL = True
except ImportError:
    HAS_MYSQL = False

MYSQL_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'root2'),
    'database': os.getenv('DB_NAME', 'deepfake_db')
}

SQLITE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deepfake_local.db')
_USE_SQLITE = False


def check_mysql_connection():
    """Verify if MySQL is accessible, creating the database if missing."""
    global _USE_SQLITE
    if not HAS_MYSQL:
        _USE_SQLITE = True
        return False
    try:
        # Check base connection
        conn = mysql.connector.connect(
            host=MYSQL_CONFIG['host'],
            user=MYSQL_CONFIG['user'],
            password=MYSQL_CONFIG['password']
        )
        cur = conn.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_CONFIG['database']}")
        cur.close()
        conn.close()
        _USE_SQLITE = False
        return True
    except Exception as e:
        print(f"[DB Notice] MySQL unavailable ({e}). Using local SQLite database.")
        _USE_SQLITE = True
        return False


def get_connection():
    """Return an active connection (MySQL or SQLite)."""
    global _USE_SQLITE
    if not _USE_SQLITE:
        try:
            conn = mysql.connector.connect(**MYSQL_CONFIG)
            return conn, 'mysql'
        except Exception:
            _USE_SQLITE = True

    # SQLite fallback
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn, 'sqlite'


def init_db():
    """Initialize database tables for users, scans, complaints, and admins."""
    check_mysql_connection()
    conn, db_type = get_connection()
    cur = conn.cursor()

    if db_type == 'mysql':
        # MySQL tables
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(64) UNIQUE NOT NULL,
            email VARCHAR(120) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            full_name VARCHAR(120),
            role VARCHAR(20) DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NULL,
            filename VARCHAR(255) NOT NULL,
            original_name VARCHAR(255) NOT NULL,
            media_type VARCHAR(20) DEFAULT 'image',
            verdict VARCHAR(20) NOT NULL,
            confidence FLOAT NOT NULL,
            fake_prob FLOAT NOT NULL,
            real_prob FLOAT NOT NULL,
            face_detected TINYINT(1) DEFAULT 1,
            face_crop_path VARCHAR(255) NULL,
            heatmap_path VARCHAR(255) NULL,
            overlay_path VARCHAR(255) NULL,
            details_json TEXT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NULL,
            scan_id INT NULL,
            name VARCHAR(100) NOT NULL,
            age VARCHAR(10),
            gender VARCHAR(20),
            mobile VARCHAR(20),
            description TEXT NOT NULL,
            image VARCHAR(255),
            status VARCHAR(30) DEFAULT 'Under Review',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
            FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE SET NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        conn.commit()

    else:
        # SQLite tables
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NULL,
            filename TEXT NOT NULL,
            original_name TEXT NOT NULL,
            media_type TEXT DEFAULT 'image',
            verdict TEXT NOT NULL,
            confidence REAL NOT NULL,
            fake_prob REAL NOT NULL,
            real_prob REAL NOT NULL,
            face_detected INTEGER DEFAULT 1,
            face_crop_path TEXT NULL,
            heatmap_path TEXT NULL,
            overlay_path TEXT NULL,
            details_json TEXT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NULL,
            scan_id INTEGER NULL,
            name TEXT NOT NULL,
            age TEXT,
            gender TEXT,
            mobile TEXT,
            description TEXT NOT NULL,
            image TEXT,
            status TEXT DEFAULT 'Under Review',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
            FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE SET NULL
        );
        """)
        conn.commit()

    cur.close()
    conn.close()

    # Seed default admin user
    seed_default_admin()


def seed_default_admin():
    """Ensure a default admin user exists."""
    from werkzeug.security import generate_password_hash
    admin = get_user_by_username('admin')
    if not admin:
        create_user(
            username='admin',
            email='admin@deepfake-forensics.gov',
            password_hash=generate_password_hash('admin123'),
            full_name='System Administrator',
            role='admin'
        )


# -------------------------------------------------------------------
# USER OPERATIONS
# -------------------------------------------------------------------
def create_user(username, email, password_hash, full_name='', role='user'):
    conn, db_type = get_connection()
    cur = conn.cursor()
    try:
        if db_type == 'mysql':
            cur.execute(
                "INSERT INTO users (username, email, password_hash, full_name, role) VALUES (%s, %s, %s, %s, %s)",
                (username, email, password_hash, full_name, role)
            )
            user_id = cur.lastrowid
        else:
            cur.execute(
                "INSERT INTO users (username, email, password_hash, full_name, role) VALUES (?, ?, ?, ?, ?)",
                (username, email, password_hash, full_name, role)
            )
            user_id = cur.lastrowid
        conn.commit()
        return user_id
    finally:
        cur.close()
        conn.close()


def get_user_by_username(username):
    conn, db_type = get_connection()
    try:
        if db_type == 'mysql':
            cur = conn.cursor(dictionary=True)
            cur.execute("SELECT * FROM users WHERE username = %s", (username,))
            res = cur.fetchone()
        else:
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE username = ?", (username,))
            row = cur.fetchone()
            res = dict(row) if row else None
        return res
    finally:
        cur.close()
        conn.close()


def get_user_by_email(email):
    conn, db_type = get_connection()
    try:
        if db_type == 'mysql':
            cur = conn.cursor(dictionary=True)
            cur.execute("SELECT * FROM users WHERE email = %s", (email,))
            res = cur.fetchone()
        else:
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE email = ?", (email,))
            row = cur.fetchone()
            res = dict(row) if row else None
        return res
    finally:
        cur.close()
        conn.close()


def get_user_by_id(user_id):
    conn, db_type = get_connection()
    try:
        if db_type == 'mysql':
            cur = conn.cursor(dictionary=True)
            cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            res = cur.fetchone()
        else:
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cur.fetchone()
            res = dict(row) if row else None
        return res
    finally:
        cur.close()
        conn.close()


def get_all_users():
    conn, db_type = get_connection()
    try:
        if db_type == 'mysql':
            cur = conn.cursor(dictionary=True)
            cur.execute("""
                SELECT u.id, u.username, u.email, u.full_name, u.role, u.created_at,
                       COUNT(s.id) as scan_count
                FROM users u
                LEFT JOIN scans s ON u.id = s.user_id
                GROUP BY u.id
                ORDER BY u.id DESC
            """)
            return cur.fetchall()
        else:
            cur = conn.cursor()
            cur.execute("""
                SELECT u.id, u.username, u.email, u.full_name, u.role, u.created_at,
                       COUNT(s.id) as scan_count
                FROM users u
                LEFT JOIN scans s ON u.id = s.user_id
                GROUP BY u.id
                ORDER BY u.id DESC
            """)
            rows = cur.fetchall()
            return [dict(r) for r in rows]
    finally:
        cur.close()
        conn.close()


def delete_user(user_id):
    conn, db_type = get_connection()
    cur = conn.cursor()
    try:
        if db_type == 'mysql':
            cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
        else:
            cur.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
    finally:
        cur.close()
        conn.close()


# -------------------------------------------------------------------
# SCAN / DETECTION HISTORY OPERATIONS
# -------------------------------------------------------------------
def create_scan(user_id, filename, original_name, media_type, verdict, confidence,
                fake_prob, real_prob, face_detected, face_crop_path, heatmap_path,
                overlay_path, details_json=None):
    conn, db_type = get_connection()
    cur = conn.cursor()
    try:
        if details_json and not isinstance(details_json, str):
            details_json = json.dumps(details_json)

        if db_type == 'mysql':
            cur.execute("""
                INSERT INTO scans (
                    user_id, filename, original_name, media_type, verdict, confidence,
                    fake_prob, real_prob, face_detected, face_crop_path, heatmap_path,
                    overlay_path, details_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                user_id, filename, original_name, media_type, verdict, confidence,
                fake_prob, real_prob, 1 if face_detected else 0,
                face_crop_path, heatmap_path, overlay_path, details_json
            ))
            scan_id = cur.lastrowid
        else:
            cur.execute("""
                INSERT INTO scans (
                    user_id, filename, original_name, media_type, verdict, confidence,
                    fake_prob, real_prob, face_detected, face_crop_path, heatmap_path,
                    overlay_path, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, filename, original_name, media_type, verdict, confidence,
                fake_prob, real_prob, 1 if face_detected else 0,
                face_crop_path, heatmap_path, overlay_path, details_json
            ))
            scan_id = cur.lastrowid
        conn.commit()
        return scan_id
    finally:
        cur.close()
        conn.close()


def get_scan_by_id(scan_id):
    conn, db_type = get_connection()
    try:
        if db_type == 'mysql':
            cur = conn.cursor(dictionary=True)
            cur.execute("""
                SELECT s.*, u.username, u.full_name, u.email
                FROM scans s
                LEFT JOIN users u ON s.user_id = u.id
                WHERE s.id = %s
            """, (scan_id,))
            res = cur.fetchone()
        else:
            cur = conn.cursor()
            cur.execute("""
                SELECT s.*, u.username, u.full_name, u.email
                FROM scans s
                LEFT JOIN users u ON s.user_id = u.id
                WHERE s.id = ?
            """, (scan_id,))
            row = cur.fetchone()
            res = dict(row) if row else None

        if res and res.get('details_json'):
            try:
                res['details'] = json.loads(res['details_json'])
            except Exception:
                res['details'] = {}
        return res
    finally:
        cur.close()
        conn.close()


def get_user_scans(user_id, limit=50):
    conn, db_type = get_connection()
    try:
        if db_type == 'mysql':
            cur = conn.cursor(dictionary=True)
            cur.execute("SELECT * FROM scans WHERE user_id = %s ORDER BY id DESC LIMIT %s", (user_id, limit))
            return cur.fetchall()
        else:
            cur = conn.cursor()
            cur.execute("SELECT * FROM scans WHERE user_id = ? ORDER BY id DESC LIMIT ?", (user_id, limit))
            rows = cur.fetchall()
            return [dict(r) for r in rows]
    finally:
        cur.close()
        conn.close()


def get_all_scans(limit=100):
    conn, db_type = get_connection()
    try:
        if db_type == 'mysql':
            cur = conn.cursor(dictionary=True)
            cur.execute("""
                SELECT s.*, u.username, u.full_name
                FROM scans s
                LEFT JOIN users u ON s.user_id = u.id
                ORDER BY s.id DESC LIMIT %s
            """, (limit,))
            return cur.fetchall()
        else:
            cur = conn.cursor()
            cur.execute("""
                SELECT s.*, u.username, u.full_name
                FROM scans s
                LEFT JOIN users u ON s.user_id = u.id
                ORDER BY s.id DESC LIMIT ?
            """, (limit,))
            rows = cur.fetchall()
            return [dict(r) for r in rows]
    finally:
        cur.close()
        conn.close()


def delete_scan(scan_id, user_id=None):
    conn, db_type = get_connection()
    cur = conn.cursor()
    try:
        if user_id:
            if db_type == 'mysql':
                cur.execute("DELETE FROM scans WHERE id = %s AND user_id = %s", (scan_id, user_id))
            else:
                cur.execute("DELETE FROM scans WHERE id = ? AND user_id = ?", (scan_id, user_id))
        else:
            if db_type == 'mysql':
                cur.execute("DELETE FROM scans WHERE id = %s", (scan_id,))
            else:
                cur.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
        conn.commit()
    finally:
        cur.close()
        conn.close()


# -------------------------------------------------------------------
# [FEATURE COMING SOON] CYBERCRIME COMPLAINTS & INCIDENT REPORTING
# NOTE: The "Report Crime" feature will be released in an upcoming update.
# The database operations below are preserved intact for future activation.
# -------------------------------------------------------------------
def create_complaint(name, age, gender, mobile, description, image=None, user_id=None, scan_id=None):
    """[FUTURE FEATURE] Insert a new citizen cybercrime complaint into the database."""
    conn, db_type = get_connection()
    cur = conn.cursor()
    try:
        if db_type == 'mysql':
            cur.execute("""
                INSERT INTO complaints (user_id, scan_id, name, age, gender, mobile, description, image)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (user_id, scan_id, name, age, gender, mobile, description, image))
            cid = cur.lastrowid
        else:
            cur.execute("""
                INSERT INTO complaints (user_id, scan_id, name, age, gender, mobile, description, image)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, scan_id, name, age, gender, mobile, description, image))
            cid = cur.lastrowid
        conn.commit()
        return cid
    finally:
        cur.close()
        conn.close()


def get_all_complaints():
    conn, db_type = get_connection()
    try:
        if db_type == 'mysql':
            cur = conn.cursor(dictionary=True)
            cur.execute("""
                SELECT c.*, u.username as user_username
                FROM complaints c
                LEFT JOIN users u ON c.user_id = u.id
                ORDER BY c.id DESC
            """)
            return cur.fetchall()
        else:
            cur = conn.cursor()
            cur.execute("""
                SELECT c.*, u.username as user_username
                FROM complaints c
                LEFT JOIN users u ON c.user_id = u.id
                ORDER BY c.id DESC
            """)
            rows = cur.fetchall()
            return [dict(r) for r in rows]
    finally:
        cur.close()
        conn.close()


def update_complaint_status(complaint_id, status):
    conn, db_type = get_connection()
    cur = conn.cursor()
    try:
        if db_type == 'mysql':
            cur.execute("UPDATE complaints SET status = %s WHERE id = %s", (status, complaint_id))
        else:
            cur.execute("UPDATE complaints SET status = ? WHERE id = ?", (status, complaint_id))
        conn.commit()
    finally:
        cur.close()
        conn.close()


def delete_complaint(complaint_id):
    conn, db_type = get_connection()
    cur = conn.cursor()
    try:
        if db_type == 'mysql':
            cur.execute("DELETE FROM complaints WHERE id = %s", (complaint_id,))
        else:
            cur.execute("DELETE FROM complaints WHERE id = ?", (complaint_id,))
        conn.commit()
    finally:
        cur.close()
        conn.close()


# -------------------------------------------------------------------
# ADMIN STATS
# -------------------------------------------------------------------
def get_admin_stats():
    conn, db_type = get_connection()
    cur = conn.cursor()
    try:
        # Total scans
        cur.execute("SELECT COUNT(*) FROM scans")
        total_scans = cur.fetchone()[0]

        # Fake detections
        cur.execute("SELECT COUNT(*) FROM scans WHERE verdict LIKE '%Fake%'")
        total_fakes = cur.fetchone()[0]

        total_reals = total_scans - total_fakes

        # Total complaints
        cur.execute("SELECT COUNT(*) FROM complaints")
        total_complaints = cur.fetchone()[0]

        # Pending complaints
        cur.execute("SELECT COUNT(*) FROM complaints WHERE status != 'Resolved'")
        pending_complaints = cur.fetchone()[0]

        # Total registered users
        cur.execute("SELECT COUNT(*) FROM users WHERE role != 'admin'")
        total_users = cur.fetchone()[0]

        fake_rate = round((total_fakes / total_scans * 100), 1) if total_scans > 0 else 0

        return {
            'total_scans': total_scans,
            'total_fakes': total_fakes,
            'total_reals': total_reals,
            'fake_rate': fake_rate,
            'total_complaints': total_complaints,
            'pending_complaints': pending_complaints,
            'total_users': total_users,
            'db_type': db_type.upper()
        }
    finally:
        cur.close()
        conn.close()
