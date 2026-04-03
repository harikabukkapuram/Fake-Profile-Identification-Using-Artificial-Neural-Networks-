-- 0001_create_tables.sql
-- Creates core tables for FakeProfileDetection app
PRAGMA foreign_keys = ON;

BEGIN TRANSACTION;

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT,
    password_hash BLOB NOT NULL,
    role TEXT NOT NULL DEFAULT 'user', -- 'user' or 'admin'
    is_blocked INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    last_login TEXT,
    notes TEXT
);

-- Flagged profiles (reported by users or auto-flagged)
CREATE TABLE IF NOT EXISTS flagged_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reported_by_user_id INTEGER, -- nullable, null => system/model flagged
    profile_id_or_handle TEXT,
    profile_raw TEXT, -- JSON text
    model_score REAL,
    model_prediction TEXT,
    status TEXT DEFAULT 'pending', -- pending/reviewed/dismissed/actioned
    admin_action TEXT, -- e.g., blocked, warning_sent
    admin_notes TEXT,
    created_at TEXT,
    reviewed_at TEXT,
    reviewed_by_admin_id INTEGER,
    FOREIGN KEY (reported_by_user_id) REFERENCES users(id),
    FOREIGN KEY (reviewed_by_admin_id) REFERENCES users(id)
);

-- Predictions log
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER, -- who ran it, nullable
    profile_raw TEXT,
    score REAL,
    prediction TEXT,
    created_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Training runs
CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_name TEXT,
    trained_by_admin_id INTEGER,
    accuracy REAL,
    precision REAL,
    recall REAL,
    f1 REAL,
    model_file TEXT,
    created_at TEXT,
    notes TEXT,
    FOREIGN KEY (trained_by_admin_id) REFERENCES users(id)
);

-- Admin logs
CREATE TABLE IF NOT EXISTS admin_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    admin_id INTEGER,
    action TEXT,
    details TEXT,
    ip TEXT,
    created_at TEXT,
    FOREIGN KEY (admin_id) REFERENCES users(id)
);

COMMIT;
