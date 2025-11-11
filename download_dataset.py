#!/usr/bin/env python3
# download_dataset.py
# Usage:
#   python download_dataset.py --test_size 0.2 --seed 42 --make_public_ids --public_frac 0.5

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

CANDIDATE_TARGETS = ["income", "class", "target"]

def find_target_column(df: pd.DataFrame, openml_target: pd.Series | None) -> pd.Series:
    # Prefer OpenML-provided target if available
    if openml_target is not None and len(openml_target) == len(df):
        return openml_target
    # Else search common names (income/class/target) case-insensitively
    lower = {c.lower(): c for c in df.columns}
    for t in CANDIDATE_TARGETS:
        if t in lower:
            return df[lower[t]]
    # Fallback: raise with column list to help debug
    raise KeyError(f"Target column not found. Available columns: {list(df.columns)}")

def binarize_income_labels(y: pd.Series) -> pd.Series:
    # Normalize strings: strip spaces, remove trailing dots, uppercase
    ys = y.astype(str).str.strip().str.replace(".", "", regex=False)
    # True if contains >50K (handles '>50K' and '>50K')
    return ys.str.contains(">50K", case=False, regex=False).astype(int)

def main():
    ap = argparse.ArgumentParser(description="Download Adult (OpenML) and create competition files (train/test/ground).")
    ap.add_argument("--out_dir", default="adult_competition_data", help="Output directory")
    ap.add_argument("--test_size", type=float, default=0.2, help="Test split proportion (default 0.2)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--make_public_ids", action="store_true", help="Also create public_ids.csv")
    ap.add_argument("--public_frac", type=float, default=0.5, help="Fraction of ground used as public (default 0.5)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("⏬ Fetching Adult dataset from OpenML (version=2)...")
    adult = fetch_openml("adult", version=2, as_frame=True)
    df = adult.frame.copy()

    # Normalize column names (keep a copy of original -> might be useful)
    df.columns = [c.strip().lower().replace("-", "_") for c in df.columns]

    # Find and binarize target
    y_raw = find_target_column(df, adult.target if hasattr(adult, "target") else None)
    y_bin = binarize_income_labels(y_raw)

    # Build a clean dataframe with binary target named 'income'
    if "income" in df.columns:
        df = df.drop(columns=["income"])
    df["income"] = y_bin

    # Stratified split
    print(f"✂️  Splitting dataset with test_size={args.test_size}, seed={args.seed} (stratified)...")
    train_df, test_df = train_test_split(
        df, test_size=args.test_size, stratify=df["income"], random_state=args.seed
    )

    # Prepare test with id and hidden ground
    test_df = test_df.reset_index(drop=True)
    test_df.insert(0, "id", range(1, len(test_df) + 1))

    # Write files
    train_csv = out_dir / "train.csv"
    test_csv = out_dir / "test.csv"
    ground_csv = out_dir / "ground.csv"
    sample_csv = out_dir / "sample_submission.csv"

    print(f"💾 Writing {train_csv}")
    train_df.to_csv(train_csv, index=False)

    print(f"💾 Writing {test_csv} (WITHOUT income)")
    test_df.drop(columns=["income"]).to_csv(test_csv, index=False)

    ground = test_df[["id", "income"]].copy()
    print(f"💾 Writing {ground_csv}")
    ground.to_csv(ground_csv, index=False)

    sample = ground[["id"]].copy()
    sample["income_prob"] = 0.5
    print(f"💾 Writing {sample_csv}")
    sample.to_csv(sample_csv, index=False)

    # Optional public/private split
    if args.make_public_ids:
        n = len(ground)
        k = int(np.floor(args.public_frac * n))
        rng = np.random.default_rng(args.seed)
        public_idx = rng.choice(ground["id"].values, size=k, replace=False)
        public_ids = pd.DataFrame({"id": np.sort(public_idx)})
        public_path = out_dir / "public_ids.csv"
        print(f"💾 Writing {public_path} (public fraction = {args.public_frac:.2f}, n={len(public_ids)})")
        public_ids.to_csv(public_path, index=False)

    print("✅ Done. Files created in:", out_dir.as_posix())

if __name__ == "__main__":
    main()
