"""
BASELINE TEMPLATE — GRU model for Wunderfund Predictorium.

The Coder agent should MODIFY this template rather than writing from scratch.
This guarantees correct data loading, evaluation, solution.py interface, and
output format.

Usage:
    python train.py --mode cv        # 3-fold CV on 20% subset
    python train.py --mode submission # train on full merged data

Output (last line of stdout): JSON with metrics.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset


# ===========================================================================
# CONFIG — Coder agent: modify these values
# ===========================================================================
CONFIG = {
    "model_type": "GRU",           # GRU | LSTM | Transformer | Mamba
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.1,
    "lr": 1e-3,
    "batch_size": 64,
    "max_epochs": 50,
    "early_stop_patience": 8,
    "early_stop_min_epochs": 5,
    "warmup_steps": 99,            # steps 0-98 not scored
    "clip_grad": 1.0,
    "weight_decay": 1e-4,
    "features": list(range(32)),   # which of the 32 features to use
    "extra_features": [],           # names of engineered features to add
    "seed": 42,
}


# ===========================================================================
# DATASET
# ===========================================================================
class PredictoriumDataset(Dataset):
    def __init__(self, parquet_path: str):
        df = pd.read_parquet(parquet_path)
        self.sequences = []
        feature_cols = [f"p{i}" for i in range(12)] + \
                       [f"v{i}" for i in range(12)] + \
                       [f"dp{i}" for i in range(4)] + \
                       [f"dv{i}" for i in range(4)]
        target_cols = ["t0", "t1"]

        for seq_ix, group in df.groupby("seq_ix"):
            group = group.sort_values("step_in_seq")
            features = group[feature_cols].values.astype(np.float32)
            targets = group[target_cols].values.astype(np.float32)
            self.sequences.append((features, targets))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        features, targets = self.sequences[idx]
        return torch.from_numpy(features), torch.from_numpy(targets)


def collate_fn(batch):
    features = torch.stack([b[0] for b in batch])  # (B, T, F)
    targets = torch.stack([b[1] for b in batch])    # (B, T, 2)
    return features, targets


# ===========================================================================
# MODEL — Coder agent: replace this class with your architecture
# ===========================================================================
class BaselineModel(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=256, num_layers=2,
                 dropout=0.1):
        super().__init__()
        self.rnn = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.rnn(x)       # (B, T, H)
        pred = self.head(out)       # (B, T, 2)
        return pred


# ===========================================================================
# EVALUATION — weighted Pearson correlation (matches Predictorium scoring)
# ===========================================================================
def weighted_pearson(y_true: np.ndarray, y_pred: np.ndarray,
                     weights: np.ndarray) -> float:
    """Weighted Pearson correlation. Handles edge cases."""
    mask = weights > 0
    if mask.sum() < 2:
        return 0.0
    y_t = y_true[mask]
    y_p = y_pred[mask]
    w = weights[mask]

    w_sum = w.sum()
    mean_t = np.average(y_t, weights=w)
    mean_p = np.average(y_p, weights=w)

    cov = np.sum(w * (y_t - mean_t) * (y_p - mean_p)) / w_sum
    std_t = np.sqrt(np.sum(w * (y_t - mean_t) ** 2) / w_sum)
    std_p = np.sqrt(np.sum(w * (y_p - mean_p) ** 2) / w_sum)

    if std_t < 1e-8 or std_p < 1e-8:
        return 0.0
    return float(cov / (std_t * std_p))


def evaluate(model, dataloader, device, warmup_steps=99):
    model.eval()
    all_preds_t0, all_preds_t1 = [], []
    all_true_t0, all_true_t1 = [], []

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)
            with autocast():
                preds = model(features)

            # Only scored steps
            preds_scored = preds[:, warmup_steps:, :].cpu().numpy()
            targets_scored = targets[:, warmup_steps:, :].cpu().numpy()

            # Clip predictions to [-6, 6]
            preds_scored = np.clip(preds_scored, -6, 6)

            # Flatten
            B, T, _ = preds_scored.shape
            all_preds_t0.append(preds_scored[:, :, 0].reshape(-1))
            all_preds_t1.append(preds_scored[:, :, 1].reshape(-1))
            all_true_t0.append(targets_scored[:, :, 0].reshape(-1))
            all_true_t1.append(targets_scored[:, :, 1].reshape(-1))

    all_preds_t0 = np.concatenate(all_preds_t0)
    all_preds_t1 = np.concatenate(all_preds_t1)
    all_true_t0 = np.concatenate(all_true_t0)
    all_true_t1 = np.concatenate(all_true_t1)

    w_t0 = np.abs(all_true_t0)
    w_t1 = np.abs(all_true_t1)

    score_t0 = weighted_pearson(all_true_t0, all_preds_t0, w_t0)
    score_t1 = weighted_pearson(all_true_t1, all_preds_t1, w_t1)
    mean_score = (score_t0 + score_t1) / 2.0

    return {"t0": score_t0, "t1": score_t1, "mean_score": mean_score}


# ===========================================================================
# LOSS — weighted MSE aligned with scoring metric
# ===========================================================================
class WeightedMSELoss(nn.Module):
    def __init__(self, warmup_steps=99):
        super().__init__()
        self.warmup_steps = warmup_steps

    def forward(self, pred, target):
        # Only score non-warmup steps
        pred = pred[:, self.warmup_steps:, :]
        target = target[:, self.warmup_steps:, :]
        weights = target.abs().clamp(min=0.01)  # avoid zero weights
        loss = (weights * (pred - target) ** 2).mean()
        return loss


# ===========================================================================
# TRAINING LOOP
# ===========================================================================
def train_fold(train_path: str, val_path: str, config: dict,
               device: torch.device, save_dir: str) -> dict:
    """Train one fold, return metrics dict."""
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    train_ds = PredictoriumDataset(train_path)
    val_ds = PredictoriumDataset(val_path)
    train_dl = DataLoader(train_ds, batch_size=config["batch_size"],
                          shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=config["batch_size"],
                        shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = BaselineModel(
        input_dim=32,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                   weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["max_epochs"]
    )
    criterion = WeightedMSELoss(warmup_steps=config["warmup_steps"])
    scaler = GradScaler()

    best_score = -1.0
    patience_counter = 0

    for epoch in range(config["max_epochs"]):
        model.train()
        epoch_loss = 0.0
        for features, targets in train_dl:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            with autocast():
                preds = model(features)
                loss = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        scheduler.step()
        metrics = evaluate(model, val_dl, device, config["warmup_steps"])

        # Early stopping
        if metrics["mean_score"] > best_score:
            best_score = metrics["mean_score"]
            patience_counter = 0
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f"{save_dir}/model_best.pt")
        else:
            patience_counter += 1

        if (epoch >= config["early_stop_min_epochs"]
                and patience_counter >= config["early_stop_patience"]):
            break

    return {"mean_score": best_score, **metrics}


# ===========================================================================
# SOLUTION.PY GENERATION
# ===========================================================================
def generate_solution_py(config: dict, model_path: str, output_path: str):
    """Generate a standalone solution.py for submission."""
    code = f'''
import numpy as np
import torch
import torch.nn as nn

class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(32, {config["hidden_dim"]}, {config["num_layers"]},
                          batch_first=True,
                          dropout={config["dropout"] if config["num_layers"] > 1 else 0.0})
        self.head = nn.Sequential(
            nn.Linear({config["hidden_dim"]}, {config["hidden_dim"] // 2}),
            nn.GELU(),
            nn.Dropout({config["dropout"]}),
            nn.Linear({config["hidden_dim"] // 2}, 2),
        )

    def forward(self, x):
        out, hidden = self.rnn(x)
        return self.head(out), hidden


class PredictionModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _Model().to(self.device)
        # Load weights from same directory as solution.py
        import os
        weight_path = os.path.join(os.path.dirname(__file__), "model_best.pt")
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.eval()
        self.hidden = None
        self._last_seq = -1

    def predict(self, data_point):
        # Reset state between sequences
        if data_point.seq_ix != self._last_seq:
            self.hidden = None
            self._last_seq = data_point.seq_ix

        x = torch.from_numpy(data_point.state).float().unsqueeze(0).unsqueeze(0)
        x = x.to(self.device)

        with torch.no_grad():
            out, self.hidden = self.model.rnn(x, self.hidden)
            pred = self.model.head(out)

        if not data_point.need_prediction:
            return None

        result = pred.squeeze().cpu().numpy()
        return np.clip(result, -6, 6)
'''
    Path(output_path).write_text(code.strip())


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cv", "submission"], default="cv")
    parser.add_argument("--experiment-dir", default=".")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = args.experiment_dir

    if args.mode == "cv":
        fold_scores = []
        for fold in range(3):
            train_path = f"data/quick_cv/fold_{fold}_train.parquet"
            val_path = f"data/quick_cv/fold_{fold}_val.parquet"
            metrics = train_fold(train_path, val_path, CONFIG, device,
                                 f"{save_dir}/fold_{fold}")
            fold_scores.append(metrics["mean_score"])
            print(f"Fold {fold}: {json.dumps(metrics)}", file=sys.stderr)

        result = {
            "mode": "cv",
            "fold_scores": fold_scores,
            "mean_score": float(np.mean(fold_scores)),
            "fold_std": float(np.std(fold_scores)),
        }
        # MUST be last line of stdout — the evaluator parses this
        print(json.dumps(result))

    elif args.mode == "submission":
        # Train on full merged data with 5% held-out sanity check
        df = pd.read_parquet("data/full_merged/merged_shuffled.parquet")
        seq_ids = df["seq_ix"].unique()
        np.random.seed(CONFIG["seed"])
        n_holdout = max(1, int(len(seq_ids) * 0.05))
        holdout_ids = np.random.choice(seq_ids, n_holdout, replace=False)
        train_ids = [s for s in seq_ids if s not in holdout_ids]

        train_df = df[df["seq_ix"].isin(train_ids)]
        holdout_df = df[df["seq_ix"].isin(holdout_ids)]

        # Save temp splits
        train_df.to_parquet(f"{save_dir}/_tmp_sub_train.parquet")
        holdout_df.to_parquet(f"{save_dir}/_tmp_sub_holdout.parquet")

        metrics = train_fold(
            f"{save_dir}/_tmp_sub_train.parquet",
            f"{save_dir}/_tmp_sub_holdout.parquet",
            CONFIG, device, save_dir,
        )

        # Generate solution.py
        generate_solution_py(CONFIG, f"{save_dir}/model_best.pt",
                             f"{save_dir}/solution.py")

        # Cleanup temp files
        Path(f"{save_dir}/_tmp_sub_train.parquet").unlink(missing_ok=True)
        Path(f"{save_dir}/_tmp_sub_holdout.parquet").unlink(missing_ok=True)

        result = {
            "mode": "submission",
            "sanity_score": metrics["mean_score"],
            **metrics,
        }
        print(json.dumps(result))


if __name__ == "__main__":
    main()
