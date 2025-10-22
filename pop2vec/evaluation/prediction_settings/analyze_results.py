#!/usr/bin/env python3
"""
analyze_results.py
==================

Utility functions for post‑processing the combined spreadsheet produced by the
960 config runs (and future ones that are still coming in).

▶  Definitions
--------------
* *task*        = unique ("task_file", "target") pair
* *task‑group*  = unique "task_file" (may contain many targets)
* *score*       = R2 for numeric rows, AUC for all others
* *model*       = the string in the "model_name" column

▶  Outputs
----------
The script writes the following CSV files into ``out_dir`` (default:
``analysis_out``):

1. ``avg_task_model.csv``              - average score for every (task, model)
2. ``avg_taskgroup_model.csv``         - average score for every (task‑group, model)
3. ``best_task_model.csv``             - best row for every (task, model)
4. ``best_taskgroup_model.csv``        - best (lr, batch) picked by mean score across all targets inside each task‑group
5. ``max_diff_task_model.csv``         - worst vs. best rows & gap for each (task, model) + row with the global max
6. ``model_ranks_best.csv``            - average rank of each model (per‑task ranking done with *best* rows)
7. ``model_ranks_avg.csv``             - same, but ranking uses *average* rows
8. ``lr_exploration_recommendations.csv`` - "explore_more" / "no" flag for each (task, model)

▶  Usage
--------
    python analyze_results.py path/to/combined_results.csv  path/to/out_dir

You can also ``import analyze_results as ar`` in a notebook and call the
individual functions directly.
"""
from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------#
# Helpers                                                                    #
# ---------------------------------------------------------------------------#
NUMERIC_TYPES = {"numeric", "ordinal", "continuous"}  # extend if needed

MetricCol = dict(
    val=dict(num="val_r2", other="val_auc"),
    test=dict(num="test_r2", other="test_auc"),
)


def _select_metric(row: pd.Series, split: str = "val") -> float:
    """Return the metric to optimise for a single row."""
    if row["type"] in NUMERIC_TYPES:
        return row[MetricCol[split]["num"]]
    return row[MetricCol[split]["other"]]


def _add_score(df: pd.DataFrame, split: str = "val") -> pd.DataFrame:
    """Attach a 'score' column with the optimised metric."""
    df = df.copy()
    df["score"] = df.apply(_select_metric, axis=1, split=split)
    return df


def _to_path(p: str | pathlib.Path) -> pathlib.Path:
    return pathlib.Path(p).expanduser().resolve()


# ---------------------------------------------------------------------------#
# 1. Average score for every (task, model)                                    #
# ---------------------------------------------------------------------------#
def average_performance_task_model(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["task_file", "target", "model_name"], dropna=False, sort=False)[
            "score"
        ]
        .mean()
        .reset_index(name="avg_score")
    )
    return g


# ---------------------------------------------------------------------------#
# 2. Average score for every (task‑group, model)                              #
# ---------------------------------------------------------------------------#
def average_performance_taskgroup_model(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["task_file", "model_name"], dropna=False, sort=False)["score"]
        .mean()
        .reset_index(name="avg_score")
    )
    return g


# ---------------------------------------------------------------------------#
# 3. Best row for every (task, model)                                         #
# ---------------------------------------------------------------------------#
def best_row_task_model(df: pd.DataFrame) -> pd.DataFrame:
    idx = (
        df.groupby(["task_file", "target", "model_name"], as_index=False, sort=False)[
            "score"
        ]
        .idxmax()
        .score
    )
    return df.loc[idx].reset_index(drop=True)


# ---------------------------------------------------------------------------#
# 4. Best (lr, batch) per (task‑group, model)                                 #
#    Strategy:                                                                #
#      . For each (task_file, model, LR, BS) compute mean score across        #
#        *all* targets inside that task‑group.                                #
#      . Pick the LR & BS that maximises that mean.                           #
# ---------------------------------------------------------------------------#
def best_lrbs_taskgroup_model(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(
            ["task_file", "model_name", "LR", "BATCH-SIZE"], dropna=False, sort=False
        )["score"]
        .mean()
        .reset_index(name="avg_score")
    )
    idx = (
        grouped.groupby(["task_file", "model_name"], as_index=False, sort=False)[
            "avg_score"
        ]
        .idxmax()
        .avg_score
    )
    return grouped.loc[idx].reset_index(drop=True)


# ---------------------------------------------------------------------------#
# 5. Gap between best & worst rows per (task, model)                          #
# ---------------------------------------------------------------------------#
@dataclass
class GapRecord:
    task_file: str
    target: str
    model_name: str
    best_idx: int
    worst_idx: int
    best_score: float
    worst_score: float
    gap: float


def max_gap_best_worst(df: pd.DataFrame) -> tuple[pd.DataFrame, GapRecord]:
    recs = []
    for keys, sub in df.groupby(["task_file", "target", "model_name"], sort=False):
        if sub.shape[0] < 2:
            continue
        best_idx = sub["score"].idxmax()
        worst_idx = sub["score"].idxmin()
        best_score = sub.loc[best_idx, "score"]
        worst_score = sub.loc[worst_idx, "score"]
        recs.append(
            {
                "task_file": keys[0],
                "target": keys[1],
                "model_name": keys[2],
                "best_idx": best_idx,
                "worst_idx": worst_idx,
                "best_score": best_score,
                "worst_score": worst_score,
                "gap": best_score - worst_score,
            }
        )
    gaps = pd.DataFrame(recs).sort_values("gap", ascending=False, ignore_index=True)
    top_row = gaps.iloc[0].to_dict()
    top = GapRecord(**top_row)
    return gaps, top


# ---------------------------------------------------------------------------#
# 6. Rank models for each task using *best* rows                              #
# ---------------------------------------------------------------------------#
def model_ranks_best(df: pd.DataFrame) -> pd.DataFrame:
    best_rows = best_row_task_model(df)
    best_rows["rank"] = (
        best_rows.groupby(["task_file", "target"], sort=False)["score"]
        .rank(method="average", ascending=False)
    )
    avg_ranks = (
        best_rows.groupby("model_name", sort=False)["rank"]
        .mean()
        .reset_index(name="avg_rank")
        .sort_values("avg_rank")
    )
    return avg_ranks


# ---------------------------------------------------------------------------#
# 7. Rank models for each task using *average* rows                           #
# ---------------------------------------------------------------------------#
def model_ranks_average(df: pd.DataFrame) -> pd.DataFrame:
    avg = average_performance_task_model(df)
    avg["rank"] = (
        avg.groupby(["task_file", "target"], sort=False)["avg_score"]
        .rank(method="average", ascending=False)
    )
    avg_ranks = (
        avg.groupby("model_name", sort=False)["rank"]
        .mean()
        .reset_index(name="avg_rank")
        .sort_values("avg_rank")
    )
    return avg_ranks


# ---------------------------------------------------------------------------#
# 8. Recommend whether to explore more LRs                                    #
# ---------------------------------------------------------------------------#
def lr_exploration_needed(df: pd.DataFrame, min_improvement: float = 0.01) -> pd.DataFrame:
    records = []
    for (task_file, target, model), sub in df.groupby(
        ["task_file", "target", "model_name"], sort=False
    ):
        # Need at least two distinct LRs
        if sub["LR"].nunique() < 2:
            continue
        sorted_sub = sub.sort_values("score", ascending=False).reset_index(drop=True)
        best = sorted_sub.iloc[0]
        second_best = sorted_sub.iloc[1]

        lr_vals = sorted_sub["LR"].unique()
        min_lr = lr_vals.min()
        max_lr = lr_vals.max()

        explore = "no"
        if (
            best["LR"] in {min_lr, max_lr}
            and best["score"] - second_best["score"] >= min_improvement
        ):
            explore = "explore_more"

        records.append(
            {
                "task_file": task_file,
                "target": target,
                "model_name": model,
                "best_lr": best["LR"],
                "best_bs": best["BATCH-SIZE"],
                "delta_vs_2nd": best["score"] - second_best["score"],
                "decision": explore,
            }
        )
    return pd.DataFrame(records)

# ---------- NEW generic rank‑count helper ----------------------------------#
def _rank_counts(df: pd.DataFrame,
                 group_cols: list[str],
                 score_col: str = "score") -> pd.DataFrame:
    ranked = df.copy()
    ranked["rank"] = (
        ranked.groupby(group_cols, sort=False)[score_col]
        .rank(method="min", ascending=False)
    )
    counts = (
        ranked.pivot_table(index="model_name",
                           columns="rank",
                           values=score_col,
                           aggfunc="count",
                           fill_value=0)
        .sort_index(axis=1)
        .reset_index()
    )
    counts.columns = ["model_name"] + [
        f"rank_{int(r)}" for r in counts.columns[1:]
    ]
    return counts


# ---------- 9. rank‑count per *task* ---------------------------------------#
def rank_counts_per_task(df: pd.DataFrame) -> pd.DataFrame:
    best_rows = best_row_task_model(df)
    return _rank_counts(best_rows, ["task_file", "target"])


# ---------- 10. rank‑count per *task‑group* --------------------------------#
def rank_counts_per_taskgroup(df: pd.DataFrame) -> pd.DataFrame:
    group_best = best_lrbs_taskgroup_model(df)
    return _rank_counts(group_best, ["task_file"], score_col="avg_score")


# ---------- 11. best / 2nd / worst per task --------------------------------#
def best_second_worst_per_task(df: pd.DataFrame) -> pd.DataFrame:
    best_rows = best_row_task_model(df)
    out = []

    for (task_file, target), sub in best_rows.groupby(["task_file", "target"],
                                                      sort=False):
        ordered = sub.sort_values("score", ascending=False,
                                  ignore_index=True)
        if ordered.shape[0] < 2:
            best = ordered.iloc[0]
            worst = ordered.iloc[-1]
            row = {
                "task_file": task_file,
                "target": target,
                "best_model": best["model_name"],
                "best_score": best["score"],
                "second_model": None,
                "second_score": None,
                "worst_model": worst["model_name"],
                "worst_score": worst["score"],
                "diff_best_second": None,
                "diff_best_worst": best["score"] - worst["score"],
            }
        else:
            best, second, worst = ordered.iloc[0], ordered.iloc[1], ordered.iloc[-1]
            row = {
                "task_file": task_file,
                "target": target,
                "best_model": best["model_name"],
                "best_score": best["score"],
                "second_model": second["model_name"],
                "second_score": second["score"],
                "worst_model": worst["model_name"],
                "worst_score": worst["score"],
                "diff_best_second": best["score"] - second["score"],
                "diff_best_worst": best["score"] - worst["score"],
            }
        out.append(row)

    return pd.DataFrame(out)


# ---------------------------------------------------------------------------#
# Helper: choose best LR-BS with full sub‑task coverage, task‑weighted       #
# ---------------------------------------------------------------------------#
def _best_complete_lrbs(sub: pd.DataFrame) -> tuple[float, int, float]:
    """
    sub - rows for ONE (task_file, model_name)  --> all its subtasks & settings

    Returns (best_lr, best_bs, best_avg_score) where best_avg_score is
    the *task‑weighted* mean:
        1. average over rows for each (LR, BS, target)
        2. then average those per‑target means across all targets

    Raises ValueError if no LR‑BS appears in **every** sub‑task.
    """
    targets = sub["target"].unique()
    n_tasks = len(targets)

    # Which LR‑BS combos cover *all* subtasks?
    coverage = (
        sub.groupby(["LR", "BATCH-SIZE"])["target"].nunique()
        .reset_index(name="n_covered")
    )
    complete = coverage.loc[coverage["n_covered"] == n_tasks, ["LR", "BATCH-SIZE"]]

    if complete.empty:
        raise ValueError(
            f"No LR‑BS tested on every sub‑task of taskgroup={sub['task_file'].iloc[0]} "
            f"for model={sub['model_name'].iloc[0]}"
        )

    # --- 1️⃣ mean score per (LR, BS, target) -------------------------------
    per_target = (
        sub.groupby(["LR", "BATCH-SIZE", "target"])["score"]
        .mean()
        .reset_index()
    )

    # --- 2️⃣ task‑weighted mean over all targets ---------------------------
    combo_means = (
        per_target.groupby(["LR", "BATCH-SIZE"])["score"]
        .mean()                            # equal weight: one vote per target
        .reset_index(name="avg_score")
    )

    # Restrict to the *complete* combos
    combo_means = combo_means.merge(complete, on=["LR", "BATCH-SIZE"])

    best_row = combo_means.loc[combo_means["avg_score"].idxmax()]
    return float(best_row["LR"]), int(best_row["BATCH-SIZE"]), float(best_row["avg_score"])


# ---------------------------------------------------------------------------#
# Updated summary_per_taskgroup_model                                        #
# ---------------------------------------------------------------------------#
def summary_per_taskgroup_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Row per (taskgroup, model) with:
         mean / min / median / max / std of the *best-per-task* scores
         best LR, BS (must exist for every sub‑task) & its task‑weighted mean
    """
    # A. stats over best‑per‑task scores
    best_rows = best_row_task_model(df)
    stats = (
        best_rows.groupby(["task_file", "model_name"], sort=False)["score"]
        .agg(["mean", "min", "median", "max", "std"])
        .reset_index()
        .rename(
            columns={
                "task_file": "taskgroup",
                "model_name": "model",
                "std": "std_deviation",
            }
        )
    )

    # B. best LR‑BS with full coverage
    rows = []
    for (taskgroup, model), sub in df.groupby(["task_file", "model_name"],
                                              sort=False):
        lr, bs, lrbs_score = _best_complete_lrbs(sub)
        rows.append(
            {
                "taskgroup": taskgroup,
                "model": model,
                "best_lr": lr,
                "best_batch_size": bs,
                "best_lr_bs_score": lrbs_score,
            }
        )
    lrbs_df = pd.DataFrame(rows)

    # C. merge and column order
    out = stats.merge(lrbs_df, on=["taskgroup", "model"], how="inner")
    return out[
        [
            "taskgroup",
            "model",
            "mean",
            "min",
            "median",
            "max",
            "std_deviation",
            "best_lr",
            "best_batch_size",
            "best_lr_bs_score",
        ]
    ]



# ---------------------------------------------------------------------------#
# Orchestration                                                              #
# ---------------------------------------------------------------------------#
def run_all(input_csv: pathlib.Path, out_dir: pathlib.Path | None = None) -> None:
    out_dir = out_dir or pathlib.Path("analysis_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    df = _add_score(df)  # adds the 'score' column

    # 1
    avg_task_model = average_performance_task_model(df)
    avg_task_model.to_csv(out_dir / "avg_task_model.csv", index=False)

    # 2
    avg_taskgroup_model = average_performance_taskgroup_model(df)
    avg_taskgroup_model.to_csv(out_dir / "avg_taskgroup_model.csv", index=False)

    # 3
    best_task_model = best_row_task_model(df)
    best_task_model.to_csv(out_dir / "best_task_model.csv", index=False)

    # 4
    best_taskgroup_model = best_lrbs_taskgroup_model(df)
    best_taskgroup_model.to_csv(out_dir / "best_taskgroup_model.csv", index=False)

    # 5
    gaps, top_gap = max_gap_best_worst(df)
    gaps.to_csv(out_dir / "max_diff_task_model.csv", index=False)
    print(
        f"[5] Largest best‑vs‑worst gap: {top_gap.gap:.3f} "
        f"(task={top_gap.task_file}/{top_gap.target}, model={top_gap.model_name})"
    )

    # 6
    ranks_best = model_ranks_best(df)
    ranks_best.to_csv(out_dir / "model_ranks_best.csv", index=False)

    # 7
    ranks_avg = model_ranks_average(df)
    ranks_avg.to_csv(out_dir / "model_ranks_avg.csv", index=False)

    # 8
    lr_recs = lr_exploration_needed(df)
    lr_recs.to_csv(out_dir / "lr_exploration_recommendations.csv", index=False)

    # 9.
    rank_task = rank_counts_per_task(df)
    rank_task.to_csv(out_dir / "rank_counts_task.csv", index=False)

    # 10.
    rank_group = rank_counts_per_taskgroup(df)
    rank_group.to_csv(out_dir / "rank_counts_taskgroup.csv", index=False)

    # 11.
    bsw = best_second_worst_per_task(df)
    bsw.to_csv(out_dir / "best_second_worst_per_task.csv", index=False)

    print(f"Additional ranking tables written to {out_dir.resolve()}")

    # 12 - one‑row summary per task‑group x model
    tg_summary = summary_per_taskgroup_model(df)
    tg_summary.to_csv(out_dir / "taskgroup_model_summary.csv", index=False)

# ---------------------------------------------------------------------------#
# CLI                                                                        #
# ---------------------------------------------------------------------------#
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyse result spreadsheet.")
    p.add_argument("csv", type=_to_path, help="Combined results CSV")
    p.add_argument(
        "out_dir",
        nargs="?",
        default="analysis_out",
        type=_to_path,
        help="Directory to write outputs (default: analysis_out/)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_all(args.csv, args.out_dir)
