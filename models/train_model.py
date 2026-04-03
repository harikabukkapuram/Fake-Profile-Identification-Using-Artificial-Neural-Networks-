# models/train_model.py
"""
Combined training script:
 - trains ANN (Keras/TensorFlow) if available and saves to models/saved/ann_model.h5
 - trains a simple PyTorch node-MLP (as GNN placeholder) if PyTorch available and saves to models/saved/gnn_model.pt
 - saves training plots and histories into models/saved/
"""
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from datetime import datetime

# plotting with headless backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# TensorFlow / Keras imports (guarded)
try:
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping
except Exception as e:
    keras = None
    layers = None
    EarlyStopping = None
    print(f"[train_model] TensorFlow not available: {e}")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROC_DIR = os.path.join(ROOT, "data", "processed")
SAVED_DIR = os.path.join(ROOT, "models", "saved")
os.makedirs(SAVED_DIR, exist_ok=True)


def _safe_stratify_split(X, y, test_size=0.15, random_state=42):
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except Exception as e:
        print(f"[train_model] stratify split failed ({e}) — falling back to regular split")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def _plot_and_save_ann_history(history, out_prefix):
    hist = history.history if hasattr(history, "history") else history
    loss = np.array(hist.get("loss", []))
    val_loss = np.array(hist.get("val_loss", []))
    acc = np.array(hist.get("accuracy", hist.get("acc", [])))
    val_acc = np.array(hist.get("val_accuracy", hist.get("val_acc", [])))

    npz_path = out_prefix + "_history.npz"
    try:
        np.savez_compressed(npz_path, loss=loss, val_loss=val_loss, acc=acc, val_acc=val_acc)
        print(f"[train_model] Saved ANN numeric history -> {npz_path}")
    except Exception as e:
        print(f"[train_model] Could not save ANN history npz: {e}")

    # loss plot
    try:
        plt.figure()
        plt.plot(loss, label="loss")
        if val_loss.size:
            plt.plot(val_loss, label="val_loss")
        plt.title("ANN Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_prefix + "_loss.png")
        plt.close()
    except Exception as e:
        print(f"[train_model] Could not plot ANN loss: {e}")

    # acc plot
    if acc.size or val_acc.size:
        try:
            plt.figure()
            if acc.size: plt.plot(acc, label="accuracy")
            if val_acc.size: plt.plot(val_acc, label="val_accuracy")
            plt.title("ANN Accuracy Curve")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_prefix + "_acc.png")
            plt.close()
        except Exception as e:
            print(f"[train_model] Could not plot ANN accuracy: {e}")


def train_ann(ann_npz="ann_dataset.npz", epochs=30, batch_size=64):
    if keras is None:
        print("[train_model] Keras not available; skipping ANN training.")
        return None

    npz_path = os.path.join(PROC_DIR, ann_npz)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"{npz_path} not found. Run feature engineering first.")

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]

    # Ensure scaler exists in SAVED_DIR (feature_engineering saves it)
    scaler_path = os.path.join(SAVED_DIR, "scaler.pkl")
    if not os.path.exists(scaler_path):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X)
        joblib.dump(scaler, scaler_path)
        print(f"[train_model] No scaler found; created & saved -> {scaler_path}")

    X_train, X_val, y_train, y_val = _safe_stratify_split(X, y, test_size=0.15, random_state=42)
    input_dim = X_train.shape[1]

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    callbacks = []
    if EarlyStopping is not None:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1))

    print(f"[train_model] Starting ANN training (epochs={epochs}, batch_size={batch_size})")
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=2, callbacks=callbacks)

    ann_path = os.path.join(SAVED_DIR, "ann_model.h5")
    model.save(ann_path)
    print(f"[train_model] ANN saved -> {ann_path}")

    out_prefix = os.path.join(SAVED_DIR, "ann_training")
    _plot_and_save_ann_history(history, out_prefix)
    return ann_path


def _plot_and_save_gnn_history(train_losses, val_losses, val_accs, out_prefix):
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    val_accs = np.array(val_accs)

    npz_path = out_prefix + "_history.npz"
    try:
        np.savez_compressed(npz_path, train_loss=train_losses, val_loss=val_losses, val_acc=val_accs)
        print(f"[train_model] Saved GNN numeric history -> {npz_path}")
    except Exception as e:
        print(f"[train_model] Could not save GNN history npz: {e}")

    try:
        plt.figure(figsize=(8, 6))

        # --- GNN Loss Curve ---
        plt.subplot(2, 1, 1)
        plt.plot(train_losses, label="train_loss")
        if val_losses.size:
            plt.plot(val_losses, label="val_loss")
        plt.title("GNN Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # --- GNN Accuracy Curve ---
        plt.subplot(2, 1, 2)
        if val_accs.size:
            plt.plot(val_accs, label="val_accuracy")
        plt.title("GNN Accuracy Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.savefig(out_prefix + "_plot.png")
        plt.close()


    except Exception as e:
        print(f"[train_model] Could not plot GNN history: {e}")


def train_gnn(gnn_pt="gnn_graph_data.pt", epochs=10, batch_size=64, lr=1e-3):
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except Exception as e:
        print(f"[train_model] PyTorch not available: {e}")
        return None

    gnn_path = os.path.join(PROC_DIR, gnn_pt)
    if not os.path.exists(gnn_path):
        print(f"[train_model] {gnn_path} not found; skipping GNN training.")
        return None

    data = torch.load(gnn_path, map_location="cpu")
    X = data["x"]  # tensor (N, F)
    y = data["y"]  # tensor (N,)

    idx = np.arange(X.shape[0])
    try:
        tr_idx, val_idx = train_test_split(idx, test_size=0.15, random_state=42, stratify=y.numpy())
    except Exception as e:
        print(f"[train_model] GNN stratify split failed ({e}) — falling back to regular split")
        tr_idx, val_idx = train_test_split(idx, test_size=0.15, random_state=42)

    X_tr = X[tr_idx]
    y_tr = y[tr_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    input_dim = X.shape[1]

    class MLP(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    tr_tensor = torch.utils.data.TensorDataset(X_tr.to(device), y_tr.to(device).float())
    val_tensor = torch.utils.data.TensorDataset(X_val.to(device), y_val.to(device).float())
    tr_loader = torch.utils.data.DataLoader(tr_tensor, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_tensor, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in tr_loader:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)
        avg_loss = total_loss / (len(tr_loader.dataset) if len(tr_loader.dataset) else 1)
        train_losses.append(avg_loss)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb)
                loss = loss_fn(out, yb)
                val_loss += float(loss.item()) * xb.size(0)
                preds = (out > 0.5).long()
                correct += (preds == yb.long()).sum().item()
                total += xb.size(0)
        val_loss_avg = val_loss / (len(val_loader.dataset) if len(val_loader.dataset) else 1)
        val_acc = correct / total if total else 0.0
        val_losses.append(val_loss_avg)
        val_accs.append(val_acc)

        print(f"[train_model][GNN] Epoch {epoch+1}/{epochs} train_loss={avg_loss:.4f} val_loss={val_loss_avg:.4f} val_acc={val_acc:.4f}")

    gnn_save = os.path.join(SAVED_DIR, "gnn_model.pt")
    try:
        torch.save(model.state_dict(), gnn_save)
        print(f"[train_model] GNN saved -> {gnn_save}")
    except Exception as e:
        print(f"[train_model] Failed to save GNN model: {e}")
        gnn_save = None

    out_prefix = os.path.join(SAVED_DIR, "gnn_training")
    _plot_and_save_gnn_history(train_losses, val_losses, val_accs, out_prefix)
    return gnn_save


if __name__ == "__main__":
    print("[train_model] Starting ANN training...")
    try:
        train_ann()
    except Exception as e:
        print(f"[train_model] ANN training failed: {e}")

    print("[train_model] Starting GNN training...")
    try:
        train_gnn()
    except Exception as e:
        print(f"[train_model] GNN training failed: {e}")
