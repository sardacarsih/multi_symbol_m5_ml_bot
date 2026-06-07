import argparse

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

from config import REPORTS_DIR
from symbols import parse_symbol_args, resolve_symbols
from train import build_model, feature_columns, fit_model, load_labeled
from utils import ensure_dirs, setup_logger

LOGGER = setup_logger("walk_forward")


def walk_forward_symbol(symbol: str) -> pd.DataFrame:
    ensure_dirs([symbol])
    df = load_labeled(symbol)
    columns = feature_columns(df)
    start = df["time"].min()
    end = df["time"].max()
    fold_start = start + pd.DateOffset(months=6)
    rows = []
    fold = 1

    while fold_start + pd.DateOffset(months=1) <= end:
        train_df = df[(df["time"] >= start) & (df["time"] < fold_start)]
        test_end = fold_start + pd.DateOffset(months=1)
        test_df = df[(df["time"] >= fold_start) & (df["time"] < test_end)]
        if len(train_df) < 100 or len(test_df) < 20:
            fold_start += pd.DateOffset(months=1)
            continue
        val_cut = int(len(train_df) * 0.85)
        train_part = train_df.iloc[:val_cut]
        val_part = train_df.iloc[val_cut:]
        model = fit_model(build_model(), train_part[columns], train_part["label"], val_part[columns], val_part["label"])
        preds = model.predict(test_df[columns])
        labels = test_df["label"]
        rows.append(
            {
                "fold": fold,
                "train_start": train_df["time"].min(),
                "train_end": train_df["time"].max(),
                "test_start": test_df["time"].min(),
                "test_end": test_df["time"].max(),
                "accuracy": accuracy_score(labels, preds),
                "precision_buy": precision_score(labels, preds, labels=[1], average="macro", zero_division=0),
                "precision_sell": precision_score(labels, preds, labels=[2], average="macro", zero_division=0),
                "recall_buy": recall_score(labels, preds, labels=[1], average="macro", zero_division=0),
                "recall_sell": recall_score(labels, preds, labels=[2], average="macro", zero_division=0),
                "total_potential_signals": int((preds > 0).sum()),
                "label_distribution": labels.value_counts().sort_index().to_dict(),
            }
        )
        fold += 1
        fold_start += pd.DateOffset(months=1)

    result = pd.DataFrame(rows)
    result.to_csv(REPORTS_DIR / symbol / "walk_forward_result.csv", index=False)
    LOGGER.info("%s walk-forward folds: %s", symbol, len(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run walk-forward validation.")
    parse_symbol_args(parser)
    args = parser.parse_args()
    for symbol in resolve_symbols(args):
        walk_forward_symbol(symbol)


if __name__ == "__main__":
    main()
