"""
Cleanup script for old flagged profiles data

This script removes or archives old flagged profiles that don't have 
model_prediction values (from the old workflow).

Usage:
    python cleanup_old_data.py [--mode delete|archive|view]
"""

import sys
import sqlite3
from pathlib import Path

# Get the database path
ROOT = Path(__file__).resolve().parent[1]
DB_PATH = ROOT / "app_data.db"

def view_old_data():
    """View old flagged profiles that will be affected"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    rows = cursor.execute("""
        SELECT 
            id,
            profile_id_or_handle,
            model_prediction,
            status,
            admin_notes,
            created_at
        FROM flagged_profiles
        WHERE model_prediction IS NULL OR model_prediction = ''
        ORDER BY created_at DESC
    """).fetchall()
    
    if not rows:
        print("✅ No old flagged profiles found. Database is clean!")
        conn.close()
        return
    
    print(f"\n📋 Found {len(rows)} old flagged profile(s):\n")
    print(f"{'ID':<5} {'Profile':<20} {'Status':<15} {'Date':<20}")
    print("-" * 70)
    
    for row in rows:
        print(f"{row['id']:<5} {row['profile_id_or_handle']:<20} {row['status']:<15} {row['created_at'][:19] if row['created_at'] else 'N/A':<20}")
    
    conn.close()
    print("\n")

def delete_old_data():
    """Delete old flagged profiles"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # First, show what will be deleted
    count = cursor.execute("""
        SELECT COUNT(*) FROM flagged_profiles
        WHERE model_prediction IS NULL OR model_prediction = ''
    """).fetchone()[0]
    
    if count == 0:
        print("✅ No old flagged profiles to delete.")
        conn.close()
        return
    
    print(f"⚠️  Warning: This will delete {count} old flagged profile(s).")
    response = input("Are you sure you want to continue? (yes/no): ")
    
    if response.lower() != 'yes':
        print("❌ Deletion cancelled.")
        conn.close()
        return
    
    cursor.execute("""
        DELETE FROM flagged_profiles 
        WHERE model_prediction IS NULL OR model_prediction = ''
    """)
    conn.commit()
    
    print(f"✅ Successfully deleted {count} old flagged profile(s).")
    conn.close()

def archive_old_data():
    """Archive old flagged profiles by marking them"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # First, show what will be archived
    count = cursor.execute("""
        SELECT COUNT(*) FROM flagged_profiles
        WHERE model_prediction IS NULL OR model_prediction = ''
    """).fetchone()[0]
    
    if count == 0:
        print("✅ No old flagged profiles to archive.")
        conn.close()
        return
    
    print(f"📦 This will archive {count} old flagged profile(s).")
    response = input("Continue? (yes/no): ")
    
    if response.lower() != 'yes':
        print("❌ Archiving cancelled.")
        conn.close()
        return
    
    cursor.execute("""
        UPDATE flagged_profiles 
        SET status = 'archived',
            admin_notes = COALESCE(admin_notes || ' | ', '') || 'Archived - old workflow'
        WHERE model_prediction IS NULL OR model_prediction = ''
    """)
    conn.commit()
    
    print(f"✅ Successfully archived {count} old flagged profile(s).")
    conn.close()

def main():
    if not DB_PATH.exists():
        print(f"❌ Error: Database not found at {DB_PATH}")
        sys.exit(1)
    
    mode = sys.argv[1] if len(sys.argv) > 1 else "view"
    mode = mode.lstrip('--')
    
    print("\n" + "="*70)
    print("   Fake Profile Detection - Old Data Cleanup")
    print("="*70)
    
    if mode == "view":
        print("\n📊 Viewing old flagged profiles...")
        view_old_data()
        print("\nℹ️  To clean up the data, run:")
        print("   python cleanup_old_data.py --delete   (to permanently delete)")
        print("   python cleanup_old_data.py --archive  (to archive/hide)")
    elif mode == "delete":
        print("\n🗑️  Delete mode")
        view_old_data()
        delete_old_data()
    elif mode == "archive":
        print("\n📦 Archive mode")
        view_old_data()
        archive_old_data()
    else:
        print(f"\n❌ Unknown mode: {mode}")
        print("Usage: python cleanup_old_data.py [--mode view|delete|archive]")
        sys.exit(1)
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

