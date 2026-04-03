# models/train_gat_te.py
"""
Train GAT-TE (Graph Attention Network with Temporal Embeddings).

Expectations:
- Node features: data/processed/ann_dataset.npz  (contains X, y, feature_cols)
- Node ids / screen_names: data/processed/features_final.csv (screen_name column aligns with ann_dataset rows)
- Edge list: datasets/edges.csv
    Required columns: source, target, timestamp
    source/target must be the same ids used in features_final.csv screen_name OR numeric indices.
    timestamp: ISO format or unix seconds. Script will try to parse common formats.

Outputs:
- models/saved/gat_te_model.pt        (model state_dict)
- models/saved/gat_te_metadata.json   (node_id -> index mapping, feature order, saved params)
- models/saved/gat_te_training.png    (loss/val_acc plot)
- models/saved/gat_te_history.npz     (train_loss/val_loss/val_acc arrays)

Notes:
- Requires torch >=1.9 and torch_geometric installed.
- CPU-friendly by default; will use GPU if available.
"""
import os
import json
import time
import math
from datetime import datetime
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROC_DIR = os.path.join(ROOT, "data", "processed")
SAVED_DIR = os.path.join(ROOT, "models", "saved")
os.makedirs(SAVED_DIR, exist_ok=True)

ANN_NPZ = os.path.join(PROC_DIR, "ann_dataset.npz")
FEATURES_CSV = os.path.join(PROC_DIR, "features_final.csv")
EDGES_CSV = os.path.join(PROC_DIR, "edges.csv")  # adjust if your edges are elsewhere

# try imports for PyG
try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GATConv, global_mean_pool
    PYG_AVAILABLE = True
except Exception as e:
    PYG_AVAILABLE = False
    _err = e

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_timestamp(ts):
    """Try to standardize a timestamp (string or numeric) -> epoch seconds (float)."""
    if ts is None or (isinstance(ts, float) and np.isnan(ts)):
        return None
    if isinstance(ts, (int, float)):
        # assume unix seconds or ms
        if ts > 1e12:  # ms
            return float(ts) / 1000.0
        return float(ts)
    s = str(ts).strip()
    # try integer
    try:
        v = float(s)
        if v > 1e12:
            return v / 1000.0
        return v
    except:
        pass
    # try common ISO formats
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.timestamp()
        except:
            pass
    # last resort, parse with pandas
    try:
        dt = pd.to_datetime(s, errors='coerce')
        if pd.isna(dt):
            return None
        return dt.astype("int64") / 1e9
    except:
        return None

class Time2Vec(nn.Module):
    """Simple time encoding (learnable) — small, used to map a scalar time -> vector."""
    def __init__(self, kernel_size=8):
        super().__init__()
        self.k = kernel_size
        self.w = nn.Parameter(torch.randn(self.k))
        self.b = nn.Parameter(torch.randn(self.k))
        self.linear_w = nn.Parameter(torch.randn(1))
        self.linear_b = nn.Parameter(torch.randn(1))
    def forward(self, t):
        # t: (N,) or (N,1) - assume float tensor seconds normalized
        if t.dim() == 1:
            t = t.unsqueeze(1)
        # linear part
        lin = self.linear_w * t + self.linear_b  # (N,1)
        sin_in = t * self.w + self.b  # (N,k)
        sin = torch.sin(sin_in)
        return torch.cat([lin, sin], dim=1)  # (N, k+1)

class GAT_TE(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, heads=4, out_dim=1, time_emb_dim=8):
        super().__init__()
        self.time2vec = Time2Vec(kernel_size=time_emb_dim-1) if time_emb_dim>1 else None
        gat_in_dim = in_dim + (time_emb_dim if self.time2vec is not None else 0)
        self.conv1 = GATConv(gat_in_dim, hidden_dim, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_dim*heads, hidden_dim, heads=1, concat=True)
        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_time_emb=None):
        # x: node features (N, F)
        # edge_index: shape [2, E] or [E, 2] (we handle both)
        # edge_time_emb: (E, T) matching edges order

        # normalize edge_index to shape [2, E]
        if edge_index.dim() == 2 and edge_index.shape[0] == 2:
            src = edge_index[0]
        elif edge_index.dim() == 2 and edge_index.shape[1] == 2:
            # transposed, convert to [2, E]
            edge_index = edge_index.t().contiguous().t()
            src = edge_index[0]
        else:
            # fallback - flatten
            src = edge_index[0].view(-1)

        # ensure src is 1D LongTensor
        src = src.long().view(-1)

        if edge_time_emb is not None:
            # make sure edge_time_emb first dim equals src length
            if edge_time_emb.size(0) != src.size(0):
                # try transpose or flatten if needed
                edge_time_emb = edge_time_emb.view(src.size(0), -1)

            N = x.size(0)
            device = x.device
            # aggregate edge_time_emb per source node by summing then dividing by counts (mean)
            Tdim = edge_time_emb.size(1)
            agg = torch.zeros((N, Tdim), device=device)
            counts = torch.zeros((N, 1), device=device)

            # vals shape (E, T)
            vals = edge_time_emb  # already (E, T)
            # index_add requires index 1D and vals matching in first dim
            # use index_add_ with proper shapes
            agg.index_add_(0, src, vals)
            counts.index_add_(0, src, torch.ones((src.size(0), 1), device=device))

            counts = counts.clamp(min=1.0)
            node_time_emb = agg / counts
            x = torch.cat([x, node_time_emb], dim=1)

        h = F.elu(self.conv1(x, edge_index))
        h = F.elu(self.conv2(h, edge_index))
        logits = torch.sigmoid(self.lin(h)).squeeze(-1)
        return logits, h

def build_graph_from_inputs():
    """Load node features and edges, return a PyG Data object and mapping."""
    if not PYG_AVAILABLE:
        raise RuntimeError(f"PyTorch Geometric not available: {_err}\nPlease install torch and torch_geometric (see README).")

    if not os.path.exists(ANN_NPZ):
        raise FileNotFoundError(f"Missing {ANN_NPZ} — run feature_engineering first.")
    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(f"Missing {FEATURES_CSV} — run feature_engineering first.")

    ann = np.load(ANN_NPZ, allow_pickle=True)
    X = ann["X"]  # (N, F)
    y = ann["y"].astype(np.int64)
    feature_cols = ann.get("feature_cols", None)
    if feature_cols is not None:
        feature_cols = list(feature_cols)

    # load node ids
    df_nodes = pd.read_csv(FEATURES_CSV, dtype=str).fillna("")
    if "screen_name" in df_nodes.columns:
        node_ids = df_nodes["screen_name"].astype(str).tolist()
    else:
        # fallback numeric indices
        node_ids = [str(i) for i in range(X.shape[0])]

    # create mapping from node id -> index (row in X)
    id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

    # load edges
    if not os.path.exists(EDGES_CSV):
        raise FileNotFoundError(f"Missing {EDGES_CSV} — please provide your edges.csv (source,target,timestamp).")
    df_e = pd.read_csv(EDGES_CSV, dtype=str).fillna("")
    # columns could be source,target,timestamp or numeric indices; allow screen_name or indices
    sources = []
    targets = []
    times = []
    for _, r in df_e.iterrows():
        s = r.get("source", "")
        t = r.get("target", "")
        ts_raw = r.get("timestamp", r.get("time", ""))
        # map to indices if possible
        if s in id_to_idx:
            s_idx = id_to_idx[s]
        else:
            try:
                s_idx = int(float(s))
            except:
                continue
        if t in id_to_idx:
            t_idx = id_to_idx[t]
        else:
            try:
                t_idx = int(float(t))
            except:
                continue
        parsed = parse_timestamp(ts_raw)
        if parsed is None:
            # set to 0 or skip - we set to 0
            parsed = 0.0
        sources.append(s_idx)
        targets.append(t_idx)
        times.append(parsed)

    if len(sources) == 0:
        raise ValueError("No valid edges loaded from edges.csv after mapping to node ids.")

    # build edge_index (undirected augmentation: add reverse edges)
    edge_index = torch.tensor([sources + targets, targets + sources], dtype=torch.long)
    # prepare edge time vector (E,) duplicated for reverse edges
    times_arr = np.array(times, dtype=float)
    times_arr = np.concatenate([times_arr, times_arr], axis=0)
    # normalize times: subtract min and divide by std
    times_norm = times_arr - times_arr.min()
    std = times_norm.std() if times_norm.std() > 0 else 1.0
    times_norm = times_norm / std
    times_tensor = torch.tensor(times_norm, dtype=torch.float32)

    # create time embeddings per edge
    # we'll use sinusoidal features as simple deterministic embedding, plus an option to learn via Time2Vec inside model
    def time_sin_cos(arr, freqs=(1.0, 0.5, 0.25, 0.125)):
        mats = []
        for f in freqs:
            mats.append(np.sin(arr * f))
            mats.append(np.cos(arr * f))
        return np.stack(mats, axis=1)  # (E, 2*len(freqs))
    edge_time_features = torch.tensor(time_sin_cos(times_norm), dtype=torch.float32)  # shape (E, T)

    # node features tensor
    x = torch.tensor(X.astype(np.float32))
    y_t = torch.tensor(y, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y_t)
    # attach edge_attr/time features on the Data object for later usage
    data.edge_time = times_tensor  # (E,)
    data.edge_time_feat = edge_time_features  # (E, T)

    meta = {
        "num_nodes": x.size(0),
        "num_node_features": x.size(1),
        "feature_cols": feature_cols if feature_cols is not None else [],
        "id_to_idx": id_to_idx
    }
    return data, meta

def train(model, data, epochs=40, batch_size=1, lr=1e-3):
    # for full-graph training we pass batch_size=1 (the whole graph)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_time_feat = data.edge_time_feat.to(device) if hasattr(data, "edge_time_feat") else None
    y = data.y.to(device).float()

    # since task is node-level binary classification, we will use masks based on random split
    N = x.size(0)
    idx = np.arange(N)
    from sklearn.model_selection import train_test_split
    tr_idx, val_idx = train_test_split(idx, test_size=0.15, stratify=data.y.numpy(), random_state=42)
    tr_idx = torch.tensor(tr_idx, dtype=torch.long, device=device)
    val_idx = torch.tensor(val_idx, dtype=torch.long, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in range(1, epochs+1):
        model.train()
        opt.zero_grad()
        out, _ = model(x, edge_index, edge_time_emb=edge_time_feat)
        loss = loss_fn(out[tr_idx], y[tr_idx])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out_eval, _ = model(x, edge_index, edge_time_emb=edge_time_feat)
            val_loss = float(loss_fn(out_eval[val_idx], y[val_idx]).item())
            # compute val acc
            preds = (out_eval[val_idx] > 0.5).long()
            correct = (preds == y[val_idx].long()).sum().item()
            acc = correct / (val_idx.size(0) if val_idx.size(0)>0 else 1.0)

        train_losses.append(loss.item())
        val_losses.append(val_loss)
        val_accs.append(acc)

        if epoch % 5 == 0 or epoch==1 or epoch==epochs:
            print(f"[GAT-TE] Epoch {epoch}/{epochs} train_loss={loss.item():.4f} val_loss={val_loss:.4f} val_acc={acc:.4f}")

    return model, (train_losses, val_losses, val_accs)

def save_artifacts(model, meta, history, prefix="gat_te"):
    model_path = os.path.join(SAVED_DIR, f"{prefix}_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[GAT-TE] Saved model -> {model_path}")

    meta_path = os.path.join(SAVED_DIR, f"{prefix}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"[GAT-TE] Saved metadata -> {meta_path}")

    # plots
    train_losses, val_losses, val_accs = history
    try:
        plt.figure(figsize=(8,6))
        plt.subplot(2,1,1)
        plt.plot(train_losses, label="train_loss")
        plt.plot(val_losses, label="val_loss")
        plt.title("GAT-TE Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(val_accs, label="val_acc")
        plt.title("GAT-TE Accuracy Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        p = os.path.join(SAVED_DIR, f"{prefix}_training.png")
        plt.savefig(p)
        plt.close()
        print(f"[GAT-TE] Saved training plot -> {p}")
    except Exception as e:
        print(f"[GAT-TE] Plotting failed: {e}")

    # numeric history
    hist_path = os.path.join(SAVED_DIR, f"{prefix}_history.npz")
    try:
        np.savez_compressed(hist_path, train_losses=np.array(train_losses), val_losses=np.array(val_losses), val_accs=np.array(val_accs))
        print(f"[GAT-TE] Saved numeric history -> {hist_path}")
    except Exception as e:
        print(f"[GAT-TE] Saving history failed: {e}")

def main(epochs=40, hidden_dim=64, heads=4, lr=1e-3, time_emb_dim=8):
    if not PYG_AVAILABLE:
        print("PyTorch Geometric (torch_geometric) not available. Please install it before running GAT-TE.")
        print("Installation (CPU example):")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
        print("  pip install torch-scatter -f https://data.pyg.org/whl/torch-<torch_version>.html")
        print("  pip install torch-sparse -f https://data.pyg.org/whl/torch-<torch_version>.html")
        print("  pip install torch-geometric -f https://data.pyg.org/whl/torch-<torch_version>.html")
        raise SystemExit(1)

    print("[GAT-TE] Building graph from inputs...")
    data, meta = build_graph_from_inputs()
    print(f"[GAT-TE] Loaded graph: nodes={data.num_nodes}, edges={data.edge_index.size(1)}")
    in_dim = data.x.size(1)
    print(f"[GAT-TE] Node feature dim: {in_dim}")

    model = GAT_TE(in_dim, hidden_dim=hidden_dim, heads=heads, time_emb_dim=time_emb_dim)
    print("[GAT-TE] Starting training...")
    model_trained, history = train(model, data, epochs=epochs, lr=lr)
    # attach meta info (timestamp, params)
    meta["trained_on"] = time.time()
    meta["params"] = {"epochs": epochs, "hidden_dim": hidden_dim, "heads": heads, "lr": lr, "time_emb_dim": time_emb_dim}
    save_artifacts(model_trained, meta, history, prefix="gat_te")
    print("[GAT-TE] Done.")

if __name__ == "__main__":
    # default run
    main()
