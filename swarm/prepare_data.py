"""
Prepare datasets for the swarm.

Creates:
  data/quick_cv/fold_{0,1,2}_{train,val}.parquet  — 30% subset, 3-fold CV
  data/full_merged/merged_shuffled.parquet         — 100% for submission training

Fix #5: Uses 30% subset (was 20%) for more reliable CV estimates.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def prepare(train_path: str = "train.parquet",
            valid_path: str = "valid.parquet",
            subset_fraction: float = 0.30,
            n_folds: int = 3,
            seed: int = 42):

    print(f"Loading training data from {train_path}...")
    train = pd.read_parquet(train_path)
    print(f"  {len(train)} rows, {train['seq_ix'].nunique()} sequences")

    print(f"Loading validation data from {valid_path}...")
    valid = pd.read_parquet(valid_path)
    print(f"  {len(valid)} rows, {valid['seq_ix'].nunique()} sequences")

    # --- Full merged shuffled dataset ---
    merged_dir = Path("data/full_merged")
    merged_dir.mkdir(parents=True, exist_ok=True)

    merged = pd.concat([train, valid]).sample(frac=1, random_state=seed)
    merged.to_parquet(merged_dir / "merged_shuffled.parquet", index=False)
    print(f"\nFull merged: {len(merged)} rows → {merged_dir / 'merged_shuffled.parquet'}")

    # --- Quick CV subset ---
    cv_dir = Path("data/quick_cv")
    cv_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    all_seq_ids = train["seq_ix"].unique()
    n_subset = int(len(all_seq_ids) * subset_fraction)
    subset_ids = rng.choice(all_seq_ids, size=n_subset, replace=False)
    subset = train[train["seq_ix"].isin(subset_ids)]

    print(f"\nCV subset: {subset_fraction*100:.0f}% = {n_subset} sequences, "
          f"{len(subset)} rows")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    subset_seq_unique = subset["seq_ix"].unique()

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(subset_seq_unique)):
        fold_train_ids = subset_seq_unique[train_idx]
        fold_val_ids = subset_seq_unique[val_idx]

        fold_train = subset[subset["seq_ix"].isin(fold_train_ids)]
        fold_val = subset[subset["seq_ix"].isin(fold_val_ids)]

        fold_train.to_parquet(cv_dir / f"fold_{fold_idx}_train.parquet", index=False)
        fold_val.to_parquet(cv_dir / f"fold_{fold_idx}_val.parquet", index=False)

        print(f"  Fold {fold_idx}: train={len(fold_train_ids)} seqs "
              f"({len(fold_train)} rows), val={len(fold_val_ids)} seqs "
              f"({len(fold_val)} rows)")

    print("\nDone. Data is ready for the swarm.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="train.parquet")
    p.add_argument("--valid", default="valid.parquet")
    p.add_argument("--subset", type=float, default=0.30)
    args = p.parse_args()
    prepare(args.train, args.valid, args.subset)
