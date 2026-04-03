# models/predictor.py
"""
Predictor module for FakeProfileDetection.

Loads available trained models (ANN, GNN placeholder, optional GAT-TE)
and exposes:
  - Predictor.predict_manual(profile_dict, screen_name=None)
  - Predictor.predict_by_index(idx)
  - Predictor.predict_by_screen_name(name)

Saves prediction rows to data/test/test_predictions.csv
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# Optional TF/Keras
try:
    from tensorflow.keras.models import load_model as keras_load_model
except Exception:
    keras_load_model = None

# Optional PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"
SAVED_DIR = ROOT / "models" / "saved"
TEST_DIR = ROOT / "data" / "test"
TEST_DIR.mkdir(parents=True, exist_ok=True)
PRED_OUT = TEST_DIR / "test_predictions.csv"

ANN_PATH = SAVED_DIR / "ann_model.h5"
SCALER_PATH = SAVED_DIR / "scaler.pkl"
GNN_PATH = SAVED_DIR / "gnn_model.pt"
GAT_PATH = SAVED_DIR / "gat_te_model.pt"            # optional (scripted/traced or state_dict)
GAT_META = SAVED_DIR / "gat_te_metadata.json"      # optional metadata for mapping
ANN_META = PROC_DIR / "ann_dataset.npz"
FEATURES_CSV = PROC_DIR / "features_final.csv"

# Lightweight MLP used if a GNN placeholder state_dict exists
if TORCH_AVAILABLE:
    class MLPNode(nn.Module):
        def __init__(self, input_dim, hidden1=128, hidden2=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden2, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

class Predictor:
    def __init__(self, weights=None):
        """
        weights: optional dict for fusion, e.g. {"ann":0.4,"gnn":0.2,"gat":0.4}
        If not provided, all available models averaged equally.
        """
        self.weights = weights or {}
        self._load_artifacts()

    def _load_artifacts(self):
        # scaler & feature_cols
        self.scaler = None
        self.feature_cols = None

        if SCALER_PATH.exists():
            try:
                self.scaler = joblib.load(str(SCALER_PATH))
            except Exception as e:
                print(f"[Predictor] Could not load scaler: {e}")

        if ANN_META.exists():
            try:
                meta = np.load(str(ANN_META), allow_pickle=True)
                if "feature_cols" in meta:
                    self.feature_cols = list(meta["feature_cols"])
            except Exception as e:
                print(f"[Predictor] Could not read ann meta: {e}")

        # fallback metadata file
        if self.feature_cols is None and (PROC_DIR / "feature_metadata.json").exists():
            try:
                mj = json.load(open(PROC_DIR / "feature_metadata.json"))
                self.feature_cols = mj.get("feature_cols")
            except:
                pass

        if self.feature_cols is None:
            # default order used across project
            self.feature_cols = [
                "statuses_count","followers_count","friends_count","favourites_count",
                "listed_count","utc_offset","description_len","has_profile_image",
                "engagement_score","follower_friend_ratio","lang_label"
            ]

        # features_final mapping (optional)
        self.features_df = None
        if FEATURES_CSV.exists():
            try:
                self.features_df = pd.read_csv(FEATURES_CSV, dtype=str).fillna("")
            except Exception as e:
                print(f"[Predictor] Could not read features_final.csv: {e}")

        # Load ANN if present
        self.ann_model = None
        if ANN_PATH.exists() and keras_load_model is not None:
            try:
                self.ann_model = keras_load_model(str(ANN_PATH))
                print("[Predictor] ANN loaded")
            except Exception as e:
                print(f"[Predictor] ANN load failed: {e}")

        # Load GNN placeholder (PyTorch MLP) if present
        self.gnn_model = None
        if TORCH_AVAILABLE and GNN_PATH.exists():
            try:
                in_dim = None
                if ANN_META.exists():
                    try:
                        m = np.load(str(ANN_META), allow_pickle=True)
                        X = m["X"]
                        in_dim = X.shape[1]
                    except:
                        in_dim = None
                if in_dim is None:
                    in_dim = len(self.feature_cols)
                m = MLPNode(in_dim)
                m.load_state_dict(torch.load(str(GNN_PATH), map_location="cpu"))
                m.eval()
                self.gnn_model = m
                self.gnn_in_dim = in_dim
                print("[Predictor] GNN placeholder loaded")
            except Exception as e:
                print(f"[Predictor] GNN placeholder load failed: {e}")

        # Try to load scripted/traced GAT-TE if it's a runnable module (optional)
        self.gat_model = None
        self.gat_meta = None
        if TORCH_AVAILABLE and GAT_META.exists():
            try:
                with open(str(GAT_META), "r") as f:
                    self.gat_meta = json.load(f)
            except Exception:
                self.gat_meta = None
        if TORCH_AVAILABLE and GAT_PATH.exists():
            try:
                # attempt torch.load; could be a scripted Module or a state_dict
                loaded = torch.load(str(GAT_PATH), map_location="cpu")
                if isinstance(loaded, torch.nn.Module):
                    loaded.eval()
                    self.gat_model = loaded
                    print("[Predictor] Loaded scripted/traced GAT-TE module.")
                else:
                    # state_dict — inference would require original class; skip direct load
                    print("[Predictor] GAT-TE is a state_dict (not a runnable module). Skipping direct load.")
            except Exception as e:
                print(f"[Predictor] GAT-TE load attempt failed: {e}")

        # Load any classical pickled models (.pkl) found in saved dir
        self.classical_models = {}
        if SAVED_DIR.exists():
            for p in SAVED_DIR.glob("*.pkl"):
                try:
                    self.classical_models[p.stem] = joblib.load(str(p))
                except Exception:
                    pass
            if self.classical_models:
                print(f"[Predictor] classical models loaded: {list(self.classical_models.keys())}")

    # ---------- vector builders ----------
    def _row_to_vector(self, row: pd.Series):
        vec = []
        for col in self.feature_cols:
            if col in row:
                try:
                    vec.append(float(row[col]))
                except:
                    vec.append(0.0)
            else:
                # derived features
                if col == "description_len":
                    vec.append(float(len(row.get("description",""))))
                elif col == "engagement_score":
                    f = float(row.get("followers_count", 0) or 0)
                    fr = float(row.get("friends_count", 0) or 0)
                    s = float(row.get("statuses_count", 0) or 0)
                    vec.append(f + fr + s)
                elif col == "follower_friend_ratio":
                    try:
                        fr = float(row.get("friends_count", 0) or 0)
                        f = float(row.get("followers_count", 0) or 0)
                        vec.append(float(f / fr) if fr > 0 else float(f))
                    except:
                        vec.append(0.0)
                elif col == "has_profile_image":
                    vec.append(1.0 if row.get("profile_image_url") else 0.0)
                else:
                    vec.append(0.0)
        arr = np.array(vec, dtype=float).reshape(1, -1)
        if self.scaler is not None:
            try:
                arr = self.scaler.transform(arr)
            except Exception:
                pass
        return arr

    def _profile_to_vector(self, profile: dict):
        # Normalizes a manual profile dict to a row-like object, then to vector
        row = {}
        keys = ["screen_name","description","followers_count","friends_count","statuses_count",
                "favourites_count","listed_count","utc_offset","profile_image_url","lang_label"]
        for k in keys:
            row[k] = profile.get(k, "")
        # coerce numeric
        for k in ["followers_count","friends_count","statuses_count","favourites_count","listed_count","utc_offset"]:
            try:
                row[k] = int(float(row.get(k, 0) or 0))
            except:
                row[k] = 0
        row["description_len"] = int(len(str(row.get("description",""))))
        row["has_profile_image"] = 1 if row.get("profile_image_url") else 0
        row["engagement_score"] = float(row["followers_count"]) + float(row["friends_count"]) + float(row["statuses_count"])
        try:
            row["follower_friend_ratio"] = float(row["followers_count"]) / float(row["friends_count"]) if row["friends_count"]>0 else float(row["followers_count"])
        except:
            row["follower_friend_ratio"] = 0.0
        if "lang_label" not in row or row["lang_label"] == "":
            row["lang_label"] = int(profile.get("lang_label", 0) or 0)
        return self._row_to_vector(pd.Series(row))

    # ---------- per-model predict helpers ----------
    def _pred_ann(self, vec):
        if self.ann_model is None:
            return None
        try:
            p = self.ann_model.predict(vec, verbose=0)
            return float(np.squeeze(p))
        except Exception as e:
            print(f"[Predictor] ANN predict error: {e}")
            return None

    def _pred_gnn(self, vec):
        if not TORCH_AVAILABLE or self.gnn_model is None:
            return None
        try:
            x = torch.tensor(vec, dtype=torch.float32)
            with torch.no_grad():
                out = self.gnn_model(x).cpu().numpy().flatten()
                return float(out[0])
        except Exception as e:
            print(f"[Predictor] GNN predict error: {e}")
            return None

    def _pred_gat(self, screen_name=None, vec=None):
        """
        If a runnable GAT-TE scripted module is available, we attempt to call it.
        Otherwise return None (GAT-TE inference requires graph/edge data & original class).
        """
        if not TORCH_AVAILABLE or self.gat_model is None:
            return None
        try:
            x = torch.tensor(vec, dtype=torch.float32)
            with torch.no_grad():
                out = self.gat_model(x)
                # handle different output shapes
                if isinstance(out, torch.Tensor):
                    return float(out.cpu().numpy().flatten()[0])
                if isinstance(out, (list, tuple)):
                    t0 = out[0]
                    if isinstance(t0, torch.Tensor):
                        return float(t0.cpu().numpy().flatten()[0])
            return None
        except Exception as e:
            print(f"[Predictor] GAT predict error: {e}")
            return None

    def _pred_classical(self, vec):
        if not self.classical_models:
            return None
        vals = []
        for name, clf in self.classical_models.items():
            try:
                if hasattr(clf, "predict_proba"):
                    prob = clf.predict_proba(vec)[0]
                    if len(prob) > 1:
                        vals.append(float(prob[1]))
                    else:
                        vals.append(float(prob[0]))
                else:
                    pred = clf.predict(vec)[0]
                    vals.append(float(pred))
            except Exception:
                pass
        return float(np.mean(vals)) if vals else None

    # ---------- public API ----------
    def predict_manual(self, profile: dict, screen_name=None):
        vec = self._profile_to_vector(profile)
        return self._predict_all(vec, screen_name=screen_name, profile=profile)

    def predict_by_screen_name(self, screen_name: str):
        if self.features_df is None:
            raise RuntimeError("features_final.csv not available")
        df = self.features_df
        if "screen_name" not in df.columns:
            raise RuntimeError("features_final.csv missing screen_name")
        match = df[df["screen_name"].astype(str) == str(screen_name)]
        if match.empty:
            match = df[df["screen_name"].astype(str).str.lower() == str(screen_name).lower()]
        if match.empty:
            raise ValueError(f"No row for {screen_name}")
        row = match.iloc[0]
        vec = self._row_to_vector(row)
        return self._predict_all(vec, screen_name=screen_name, profile=row.to_dict())

    def predict_by_index(self, idx: int):
        if self.features_df is None:
            raise RuntimeError("features_final.csv not available")
        df = self.features_df
        if idx < 0 or idx >= len(df):
            raise IndexError("index out of range")
        row = df.iloc[idx]
        vec = self._row_to_vector(row)
        return self._predict_all(vec, screen_name=str(row.get("screen_name","")), profile=row.to_dict())

    def _predict_all(self, vec, screen_name=None, profile=None):
        """
        Run available models, fuse outputs, compute label & confidence metrics.
        - final_score: probability of fake
        - label: 'fake' if final_score >= 0.5 else 'genuine'
        - label_confidence: confidence in predicted label (percentage)
        - certainty: distance from 0.5 mapped to 0..100
        """
        ann_score = self._pred_ann(vec) if getattr(self, "ann_model", None) is not None else None
        gnn_score = self._pred_gnn(vec) if getattr(self, "gnn_model", None) is not None else None
        gat_score = None
        if getattr(self, "gat_model", None) is not None:
            try:
                gat_score = self._pred_gat(screen_name=screen_name, vec=vec)
            except Exception as e:
                print(f"[Predictor] GAT predict error: {e}")
                gat_score = None
        classical_score = self._pred_classical(vec) if getattr(self, "classical_models", None) else None

        available = {}
        if ann_score is not None: available["ann"] = float(ann_score)
        if gnn_score is not None: available["gnn"] = float(gnn_score)
        if gat_score is not None: available["gat"] = float(gat_score)
        if classical_score is not None: available["classical"] = float(classical_score)

        if not available:
            final_score = 0.5
        else:
            if self.weights:
                s = 0.0; wsum = 0.0
                for k, v in available.items():
                    w = float(self.weights.get(k, 1.0))
                    s += v * w
                    wsum += w
                final_score = float(s / wsum) if wsum > 0 else float(np.mean(list(available.values())))
            else:
                final_score = float(np.mean(list(available.values())))

        label = "fake" if final_score >= 0.5 else "genuine"
        if label == "genuine":
            label_confidence = (1.0 - final_score) * 100.0
        else:
            label_confidence = final_score * 100.0

        certainty = abs(final_score - 0.5) / 0.5 * 100.0

        out = {
            "screen_name": screen_name or (profile.get("screen_name","") if profile else ""),
            "ann_score": ann_score,
            "gnn_score": gnn_score,
            "gat_score": gat_score,
            "classical_score": classical_score,
            "final_score": final_score,
            "label": label,
            "label_confidence": round(label_confidence, 6),
            "certainty": round(certainty, 6),
            "timestamp": datetime.utcnow().isoformat()
        }

        # persist
        try:
            df_new = pd.DataFrame([out])
            if PRED_OUT.exists():
                df_old = pd.read_csv(PRED_OUT)
                df_all = pd.concat([df_old, df_new], ignore_index=True)
                df_all.to_csv(PRED_OUT, index=False)
            else:
                df_new.to_csv(PRED_OUT, index=False)
            # friendly print
            print(f"[Predictor] saved prediction to {PRED_OUT}")
        except Exception as e:
            print(f"[Predictor] Could not save prediction: {e}")

        return out
