# preprocessing/preprocess.py
"""
Step 1: Load raw CSVs (fakeusers_raw.csv, users_raw.csv), label them,
clean basic issues, merge into one CSV: data/processed/merged_clean.csv

Updates:
- Uses pathlib for paths
- Safer CSV loading (encoding + engine fallback)
- Normalizes created_at to ISO when possible
- Adds derived features expected by feature_engineering/predict:
    description_len, has_profile_image, engagement_score, follower_friend_ratio
- Deduplicates on screen_name, trims whitespace
- Better logging
"""

from pathlib import Path
from datetime import datetime
import os
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DT_FORMATS = [
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%d-%b-%Y",
    "%a %b %d %H:%M:%S %z %Y",
]

def load_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[preprocess] Missing file: {path}")
        return pd.DataFrame()
    for enc in ("utf-8", "latin1", "utf-8-sig"):
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[''], encoding=enc, engine="python")
        except Exception as e:
            print(f"[preprocess] read {path} with encoding={enc} failed: {e}")
    # final attempt with pandas default
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[''])
    except Exception as e:
        print(f"[preprocess] final read failed for {path}: {e}")
        return pd.DataFrame()

def parse_created_at(val: str) -> str:
    if not isinstance(val, str) or val.strip() == "":
        return ""
    s = val.strip()
    # Try pandas to_datetime first (flexible)
    try:
        ts = pd.to_datetime(s, errors="coerce", utc=True)
        if not pd.isna(ts):
            return ts.isoformat()
    except Exception:
        pass
    # Fallback attempts using known formats
    for fmt in DEFAULT_DT_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
            # convert to ISO (no tz info unless present)
            return dt.isoformat()
        except Exception:
            continue
    return s  # give back original if we couldn't parse

def cast_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        else:
            df[c] = 0
    return df

def derive_basic_fields(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure string columns exist
    for col in ["description", "profile_image_url", "screen_name", "name", "location", "lang"]:
        if col not in df.columns:
            df[col] = ""
    # description length
    df["description_len"] = df["description"].fillna("").astype(str).apply(len)
    # has_profile_image flag (1 if url non-empty)
    df["has_profile_image"] = df["profile_image_url"].fillna("").astype(str).apply(lambda x: 1 if str(x).strip() else 0)
    # engagement_score = followers + friends + statuses
    df["engagement_score"] = (
        df.get("followers_count", 0).fillna(0).astype(int) +
        df.get("friends_count", 0).fillna(0).astype(int) +
        df.get("statuses_count", 0).fillna(0).astype(int)
    )
    # follower_friend_ratio safe
    def safe_ratio(row):
        fr = int(row.get("friends_count", 0) or 0)
        fo = int(row.get("followers_count", 0) or 0)
        try:
            return fo / fr if fr > 0 else fo
        except Exception:
            return 0
    df["follower_friend_ratio"] = df.apply(safe_ratio, axis=1)
    return df

def run_preprocess(fake_csv="fakeusers_raw.csv", real_csv="users_raw.csv"):
    fake_path = RAW_DIR / fake_csv
    real_path = RAW_DIR / real_csv

    df_fake = load_csv_safe(fake_path)
    df_real = load_csv_safe(real_path)

    # Add label column: fake = 1, real = 0
    if not df_fake.empty:
        df_fake["label"] = 1
    if not df_real.empty:
        df_real["label"] = 0

    # Ensure created_at exists and normalize
    for df in (df_fake, df_real):
        if "created_at" not in df.columns:
            df["created_at"] = ""
        else:
            df["created_at"] = df["created_at"].fillna("").astype(str).apply(parse_created_at)

    # Standardize numeric columns we care about
    numeric_cols = [
        "statuses_count", "followers_count", "friends_count", "favourites_count",
        "listed_count", "utc_offset"
    ]
    df_fake = cast_numeric(df_fake, numeric_cols)
    df_real = cast_numeric(df_real, numeric_cols)

    # Ensure important string columns exist (again)
    for col in ["description", "profile_image_url", "screen_name", "name", "location", "lang"]:
        if col not in df_fake.columns:
            df_fake[col] = ""
        if col not in df_real.columns:
            df_real[col] = ""

    # Combine
    combined = pd.concat([df_fake, df_real], ignore_index=True, sort=False).reset_index(drop=True)

    # Basic cleaning: drop rows without screen_name
    combined["screen_name"] = combined["screen_name"].fillna("").astype(str).str.strip()
    combined = combined[combined["screen_name"] != ""].copy()

    # Trim whitespace for string columns
    str_cols = combined.select_dtypes(include=["object"]).columns.tolist()
    for c in str_cols:
        combined[c] = combined[c].astype(str).str.strip()

    # Deduplicate by screen_name (keep first)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["screen_name"], keep="first").reset_index(drop=True)
    after = len(combined)
    if before != after:
        print(f"[preprocess] Dropped {before-after} duplicate screen_name rows")

    # Fill NaNs
    combined = combined.fillna("")

    # Derive features used downstream
    combined = derive_basic_fields(combined)

    # Ensure consistent types for a few columns
    for c in ["description_len", "has_profile_image", "engagement_score"]:
        if c in combined.columns:
            combined[c] = pd.to_numeric(combined[c], errors="coerce").fillna(0).astype(int)

    out_path = PROC_DIR / "merged_clean.csv"
    combined.to_csv(out_path, index=False)
    print(f"[preprocess] Wrote merged_clean.csv -> {out_path}")
    return str(out_path)

if __name__ == "__main__":
    run_preprocess()
