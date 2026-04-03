# features/feature_engineering.py
"""
Feature engineering step.
Reads data/processed/merged_clean.csv and emits:
 - data/processed/features_final.csv
 - data/processed/ann_dataset.npz   (X, y, feature_cols)
 - models/saved/scaler.pkl
 - models/saved/label_encoder.pkl
 - data/processed/gnn_graph_data.pt (if torch available)
This script is defensive: if input missing, it logs and exits gracefully.
"""
import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROC_DIR = os.path.join(ROOT, "data", "processed")
SAVED_DIR = os.path.join(ROOT, "models", "saved")
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(SAVED_DIR, exist_ok=True)

INPUT_CSV = os.path.join(PROC_DIR, "merged_clean.csv")

# --- additional helpers for edges generation/normalization ---
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

# datasets dir and edges output path
DATASETS_DIR = os.path.join(ROOT, "data", "processed")
os.makedirs(DATASETS_DIR, exist_ok=True)
OUT_EDGES_PATH = os.path.join(DATASETS_DIR, "edges.csv")


def _try_load_raw_edges():
    """
    Look for a raw interactions/edges file commonly named in projects.
    Returns path or None.
    """
    candidates = [
        os.path.join(ROOT, "datasets", "edges_raw.csv"),
        os.path.join(ROOT, "data", "raw", "interactions.csv"),
        os.path.join(ROOT, "datasets", "edges_raw.csv"),
        os.path.join(ROOT, "data", "raw", "edges.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def generate_synthetic_edges(node_ids, out_path=OUT_EDGES_PATH, avg_degree=3, temporal_span_days=365, seed=42):
    """
    Create a synthetic edges.csv with columns: source,target,timestamp
    - node_ids: list of screen_name strings (must match features_final order)
    - avg_degree: avg outgoing edges per node (Poisson param)
    - temporal_span_days: create timestamps uniformly within last N days
    """
    random.seed(seed)
    np.random.seed(seed)
    N = len(node_ids)
    edges = []
    now = datetime.utcnow()
    start = now - timedelta(days=max(1, int(temporal_span_days)))

    for i, src in enumerate(node_ids):
        # sample number of outgoing edges (avoid zero for some nodes)
        k = np.random.poisson(lam=max(1, avg_degree))
        # limit k to N-1
        k = max(0, min(k, N - 1))
        if k == 0:
            continue
        # sample targets without self
        choices = list(range(N))
        choices.remove(i)
        targets = random.sample(choices, k if k <= len(choices) else len(choices))
        for t in targets:
            ts = start + timedelta(seconds=random.randint(0, int((now - start).total_seconds())))
            # ISO format without timezone (parsable by train_gat_te)
            edges.append((str(src), str(node_ids[t]), ts.strftime("%Y-%m-%d %H:%M:%S")))

    # Save CSV
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source", "target", "timestamp"])
        writer.writerows(edges)

    print(f"[feature_engineering] Wrote synthetic edges -> {out_path} (nodes={N}, edges={len(edges)})")
    return out_path


def normalize_raw_edges_to_dataset(raw_path, node_ids, out_path=OUT_EDGES_PATH, id_map_by_name=True):
    """
    Normalize a raw edges file to (source,target,timestamp) where source/target are screen_name or numeric indices.
    If raw edges refer to screen_name, keep them; otherwise map numeric indices.
    node_ids: list of screen_name (ordered as features_final)
    """
    df_raw = None
    try:
        df_raw = pd.read_csv(raw_path, dtype=str).fillna("")
    except Exception as e:
        print(f"[feature_engineering] Could not read raw edges {raw_path}: {e}")
        return None

    # prefer columns named source/target/timestamp
    cols = [c.lower() for c in df_raw.columns]
    # try to find columns
    def find_col(poss):
        for p in poss:
            if p in cols:
                return df_raw.columns[cols.index(p)]
        return None

    s_col = find_col(["source", "src", "from"])
    t_col = find_col(["target", "tgt", "to"])
    time_col = find_col(["timestamp", "time", "ts", "created_at", "date"])

    if s_col is None or t_col is None:
        print(f"[feature_engineering] Raw edges at {raw_path} missing source/target columns. Skipping normalization.")
        return None

    out_rows = []
    id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    for _, r in df_raw.iterrows():
        s = str(r.get(s_col, "")).strip()
        t = str(r.get(t_col, "")).strip()
        ts_raw = r.get(time_col, "") if time_col else ""
        # try mapping s/t to known node ids
        if s in id_to_idx and t in id_to_idx:
            out_rows.append((s, t, ts_raw if ts_raw else "0"))
        else:
            # try numeric indices
            try:
                s_idx = int(float(s))
                t_idx = int(float(t))
                if 0 <= s_idx < len(node_ids) and 0 <= t_idx < len(node_ids):
                    out_rows.append((str(node_ids[s_idx]), str(node_ids[t_idx]), ts_raw if ts_raw else "0"))
            except:
                # skip if unmapped
                continue

    if not out_rows:
        print(f"[feature_engineering] No valid normalized rows found in {raw_path}.")
        return None

    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source", "target", "timestamp"])
        writer.writerows(out_rows)

    print(f"[feature_engineering] Normalized raw edges -> {out_path} (rows={len(out_rows)})")
    return out_path

# --- end helpers ---


class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        # Base numeric features we want (deterministic order)
        self.numeric_features = [
            "statuses_count", "followers_count", "friends_count", "favourites_count",
            "listed_count", "utc_offset", "description_len", "has_profile_image"
        ]
        # Final deterministic ANN order (append engineered features in order)
        self.ann_order = self.numeric_features + ["engagement_score", "follower_friend_ratio"]
        # metadata file to save
        self.meta_path = os.path.join(SAVED_DIR, "feature_metadata.json")

    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # description length
        df["description"] = df.get("description", "").astype(str)
        df["description_len"] = df["description"].apply(len)
        # has profile image flag
        df["profile_image_url"] = df.get("profile_image_url", "")
        df["has_profile_image"] = df["profile_image_url"].astype(bool).astype(int)
        # engagement score (simple): followers + friends + statuses
        df["engagement_score"] = (
            df["followers_count"].fillna(0).astype(int) +
            df["friends_count"].fillna(0).astype(int) +
            df["statuses_count"].fillna(0).astype(int)
        )
        # follower/friend ratio (safe)
        def ratio(r):
            try:
                fr = int(r["friends_count"])
                if fr > 0:
                    return float(r["followers_count"]) / float(fr)
                else:
                    return float(r["followers_count"])
            except Exception:
                return 0.0
        df["follower_friend_ratio"] = df.apply(ratio, axis=1)
        # language clean
        df["lang_clean"] = df.get("lang", "").fillna("").astype(str)
        return df

    def fit_transform(self, merged_csv="merged_clean.csv"):
        path = os.path.join(PROC_DIR, merged_csv)
        if not os.path.exists(path):
            print(f"[feature_engineering] Input file not found: {path}. Please run preprocessing.")
            return None

        # read, allow messy strings; we'll coerce numeric columns below
        df = pd.read_csv(path, dtype=str).fillna("")

        # cast numeric columns safely
        numeric_cols = ["statuses_count", "followers_count", "friends_count", "favourites_count", "listed_count", "utc_offset"]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0).astype(int)

        df = self._create_basic_features(df)

        # Label encode language (fit on available data)
        try:
            le_vals = df["lang_clean"].astype(str).fillna("").values
            # stable sorting of classes
            self.label_encoder.fit(le_vals)
            df["lang_label"] = self.label_encoder.transform(le_vals)
        except Exception as e:
            print(f"[feature_engineering] Warning: LabelEncoder failed: {e}")
            df["lang_label"] = 0

        # Ensure deterministic feature column ordering
        keep_cols = list(self.ann_order) + ["lang_label"]
        # if any of these are missing, add them with zeros
        for c in keep_cols:
            if c not in df.columns:
                df[c] = 0

        # Create features_final
        if "screen_name" not in df.columns:
            df["screen_name"] = df.index.astype(str)
        if "label" not in df.columns:
            # if label missing, create zeros (unlabeled)
            df["label"] = 0

        features_final = df[["screen_name", "label"] + keep_cols].copy()
        features_final_path = os.path.join(PROC_DIR, "features_final.csv")
        features_final.to_csv(features_final_path, index=False)
        print(f"[feature_engineering] Wrote features_final.csv -> {features_final_path}")

        # Prepare ANN dataset (X, y)
        X = features_final[keep_cols].values.astype(float)
        y = features_final["label"].fillna(0).astype(int).values

        # Fit scaler and save
        X_scaled = self.scaler.fit_transform(X)
        ann_npz_path = os.path.join(PROC_DIR, "ann_dataset.npz")
        # Save feature_cols as JSON-safe list
        np.savez_compressed(ann_npz_path, X=X_scaled, y=y, feature_cols=np.array(keep_cols, dtype=object))
        print(f"[feature_engineering] Wrote ann_dataset.npz -> {ann_npz_path}")

        scaler_path = os.path.join(SAVED_DIR, "scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        le_path = os.path.join(SAVED_DIR, "label_encoder.pkl")
        joblib.dump(self.label_encoder, le_path)
        print(f"[feature_engineering] Saved scaler -> {scaler_path}")
        print(f"[feature_engineering] Saved label encoder -> {le_path}")

        # Save metadata json (feature order, counts)
        meta = {
            "feature_cols": keep_cols,
            "num_rows": int(len(features_final)),
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f)
        print(f"[feature_engineering] Saved metadata -> {self.meta_path}")

        # ---------- ensure edges.csv exists for graph models ----------
        try:
            node_ids = features_final["screen_name"].astype(str).tolist()
            raw_edges = _try_load_raw_edges()
            if raw_edges:
                normalized = normalize_raw_edges_to_dataset(raw_edges, node_ids, out_path=OUT_EDGES_PATH)
                if normalized is None:
                    generate_synthetic_edges(node_ids, out_path=OUT_EDGES_PATH, avg_degree=3, temporal_span_days=365, seed=42)
            else:
                generate_synthetic_edges(node_ids, out_path=OUT_EDGES_PATH, avg_degree=3, temporal_span_days=365, seed=42)
        except Exception as e:
            print(f"[feature_engineering] edges generation/normalization failed: {e}")

        # Optionally save torch node features if available
        try:
            import torch
            node_features = torch.tensor(X_scaled, dtype=torch.float32)
            labels = torch.tensor(y, dtype=torch.long)
            gnn_path = os.path.join(PROC_DIR, "gnn_graph_data.pt")
            torch.save({"x": node_features, "y": labels}, gnn_path)
            print(f"[feature_engineering] Wrote gnn_graph_data.pt -> {gnn_path}")
        except Exception as e:
            print(f"[feature_engineering] Skipped saving torch gnn data (torch not available): {e}")

        return features_final_path


if __name__ == "__main__":
    fe = FeatureEngineer()
    fe.fit_transform()
