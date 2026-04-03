#!/usr/bin/env python3
"""
Run SQLite migrations safely:
- backs up the database file
- applies SQL files in this directory in filename order
"""

import os
import sqlite3
import shutil
from glob import glob
from datetime import datetime

# Adjust these paths if your repo structure differs
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIGRATIONS_DIR = os.path.join(ROOT, "migrations")
DB_PATH = os.path.join(ROOT, "app_data.db")   # change if your DB is named differently

def backup_db(db_path):
    if not os.path.exists(db_path):
        print(f"No DB found at {db_path}. A new DB will be created.")
        return None
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    backup_path = f"{db_path}.backup.{ts}"
    shutil.copy2(db_path, backup_path)
    print(f"Backed up DB to: {backup_path}")
    return backup_path

def apply_sql_file(conn, sql_path):
    print(f"Applying {sql_path} ...")
    with open(sql_path, "r", encoding="utf-8") as fh:
        sql = fh.read()
    try:
        conn.executescript(sql)
        print(f"Applied: {os.path.basename(sql_path)}")
    except sqlite3.Error as e:
        print(f"ERROR applying {sql_path}: {e}")
        raise

def main():
    # find SQL files
    sql_files = sorted(glob(os.path.join(MIGRATIONS_DIR, "*.sql")))
    if not sql_files:
        print("No migration .sql files found in", MIGRATIONS_DIR)
        return

    backup_db(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    try:
        for f in sql_files:
            apply_sql_file(conn, f)
        print("All migrations applied successfully.")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
