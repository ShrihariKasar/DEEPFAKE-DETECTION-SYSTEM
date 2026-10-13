-- ==========================================================
-- Ai Based Synthetic Media Detection System - DATABASE SCHEMA
-- Deepfake Detection & Incident Management System
-- ==========================================================

CREATE DATABASE IF NOT EXISTS deepfake_db; 
USE deepfake_db;

-- 1. Users Table (Administrators & Forensic Analysts)
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(64) UNIQUE NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(120),
    role VARCHAR(20) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Seed Default Administrator (admin / admin123)
-- Password hash: pbkdf2:sha256 for 'admin123'
INSERT IGNORE INTO users (username, email, password_hash, full_name, role)
VALUES (
    'admin',
    'admin@deepfake-forensics.gov',
    'pbkdf2:sha256:260000$sQj9uYjK5i1e2a3b$3a51fef3e5d0b4ec2b814dfdfa4c0a52033bc68c07c126ec001859d04c1c28c8',
    'System Administrator',
    'admin'
);

-- 2. Scans & Detection Audit Trail Table
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

-- 3. Cybercrime Complaints Table
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