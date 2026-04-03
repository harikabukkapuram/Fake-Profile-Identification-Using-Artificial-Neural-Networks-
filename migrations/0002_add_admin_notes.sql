-- Migration: Add admin_notes and admin_action columns to flagged_profiles
-- This migration is safe to run multiple times (idempotent)

-- Add admin_notes column if it doesn't exist
-- SQLite doesn't support IF NOT EXISTS for ALTER TABLE, so check manually or use Python migration
-- This SQL is for reference; Python migration in app.py handles it safely

ALTER TABLE flagged_profiles ADD COLUMN admin_notes TEXT;
ALTER TABLE flagged_profiles ADD COLUMN admin_action TEXT;

