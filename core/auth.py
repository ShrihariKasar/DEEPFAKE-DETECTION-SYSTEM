"""
Authentication and Session Security Module.
Provides secure password hashing, session checking decorators, and credential validation.
"""

from functools import wraps
from flask import session, redirect, url_for, flash, request
from werkzeug.security import generate_password_hash, check_password_hash
import re
import db


def hash_password(password: str) -> str:
    """Generate secure hash using PBKDF2 with SHA-256."""
    return generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)


def verify_password(password: str, password_hash: str) -> bool:
    """Verify raw password against hash."""
    return check_password_hash(password_hash, password)


def validate_registration(username, email, password, confirm_password):
    """Validate registration parameters. Returns (is_valid, error_message)."""
    if not username or len(username.strip()) < 3:
        return False, "Username must be at least 3 characters long."
    if not re.match("^[a-zA-Z0-9_.-]+$", username):
        return False, "Username can only contain letters, numbers, dots, dashes, and underscores."
    
    email_regex = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    if not email or not re.match(email_regex, email):
        return False, "Please provide a valid email address."

    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters long."

    if password != confirm_password:
        return False, "Passwords do not match."

    if db.get_user_by_username(username):
        return False, f"Username '{username}' is already taken."

    if db.get_user_by_email(email):
        return False, f"An account with email '{email}' already exists."

    return True, ""


def login_required(f):
    """Decorator to protect routes that require standard user authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Authentication required. Please sign in to proceed.", "warning")
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """Decorator to protect administrative routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('role') != 'admin' and not session.get('admin'):
            flash("Restricted access: Administrator clearance required.", "danger")
            return redirect(url_for('admin_login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function


def get_current_user():
    """Retrieve the currently logged-in user dictionary from DB, if any."""
    user_id = session.get('user_id')
    if user_id:
        return db.get_user_by_id(user_id)
    return None
