#!/usr/bin/env python3
"""
config_generator.py
===================

Generate training‑mode configuration **JSON** files for every combination of
MODEL_NAMES x TASKS x LR x BATCH_SIZE.

The script also emits two index artefacts - a YAML list and a CSV table - so you
can iterate over the runs easily (e.g. when generating Slurm batch files later).

Edit the global variables ROOT, RESULT_DIR, MODEL_SAVE_DIR and TASKS to match
your environment.
"""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import yaml  # PyYAML

import sys
import copy

import pandas as pd

# ---------------------------------------------------------------------------
# User‑tunable globals -------------------------------------------------------
# NOTE: Leave them as "?" placeholders for now; swap in real paths later.
# ---------------------------------------------------------------------------
ROOT = Path("/gpfs/ostor/ossc9424/data/evaluation_july25/data/subset-splits/")   
SEQ_FULL_POP = "/gpfs/ostor/ossc9424/data/llm/runs-pipeline-steps/D3-full-pop/step-5/encoding=nomlm/masking=random/encoded.h5"
SEQ_OLD = "/gpfs/ostor/ossc9424/data/llm/runs-pipeline-steps/D3-parents-siblings/step-5/encoding=nomlm/masking=random/encoded.h5"
RESULT_DIR = Path(ROOT.parent.parent, "fine-tuning_results_29-07-25")    # Root for training outputs / metrics
MODEL_NAMES = [
    "medium-random",
    "BASE-random",
    "BASE-event",
    "cceff-random",
    "ccall-random",
]

BATCH_SIZE = 32

TASKS = [
    # Populate the remaining tasks by extending this list. Add any optional
    # fields your training code understands; they will be passed through.
    # {
    #     "task_file": "twin",
    #     "target_column": {"is_twin": ["binary", 1]},  # example; replace with real target(s)
    #     "PRIMARY_KEY": "RINPERSOON1",
    #     "PARTNER_KEY": "RINPERSOON2",
    # },
    {
        "task_file": "preFer",
        "target_column": {"children_post2021": ["binary", 2]},  # example; replace with real target
    },
    {
        "task_file": "partnership-after-2020",
        "target_column": {"first_union_after_2020": ["binary", 2]},  # example; replace with real target(s)
    },
    # {
    #     "task_file": "INPA2023TABV1",
    #     "target_column": {
    #         # "INPBELI": ['numeric', 1],
    #         # "INPPG710PEN": ['numeric', 1],
    #         # "INPPH780OUV": ['numeric', 1],
    #         # "INPT5280PEN": ['numeric', 1],
    #         "SOCIAL_SECURITY": ['numeric', 1],
    #     },  # example; replace with real target(s)
    # },
    # {
    #     "task_file": "INPA2022TABV3",
    #     "target_column": {
    #         # "INPBELI": ['numeric', 1],
    #         # "INPPG710PEN": ['numeric', 1],
    #         "INPPH780OUV": ['numeric', 1],
    #         # "INPT5280PEN": ['numeric', 1],
    #         "SOCIAL_SECURITY": ['numeric', 1],
    #     },  # example; replace with real target(s)
    # },
    # # {
    # #     "task_file": "divorce-couple-after-2020",
    # #     "target_column": {"divorce_after_2020": ["binary", 2]},  # example; replace with real target(s)
    # # },
    {
        "task_file": "divorce-after-2020",
        "target_column": {"divorce_after_2020": ["binary", 2]},  # example; replace with real target(s)
        # "PARTNER_KEY": "RINPERSOONVERBINTENISP",
    },
    # {
    #     "task_file": "liss_filtered",
    #     "target_column": {
    #         # "cr030": ["categorical", 6],
    #         # "cr166": ["binary", 2],
    #         # "ca008": ["binary", 2],
    #         # "cs039": ["binary", 2],
    #         # "ch178": ["binary", 2],
    #         # "cv309": ["categorical", 14],
    #         # "cr092": ["ordinal", 4],
    #         # "cp0101": ["ordinal", 10],
    #         # "cp076": ["ordinal", 7],
    #         # "cp201": ["ordinal", 5],
    #         # "cp073": ["ordinal", 7],
    #         # "cv109": ["ordinal", 5],
    #         # "cv120": ["ordinal", 5],
    #         "cv246": ["numeric", 1],
    #         # "cv248": ["numeric", 1],
    #         # "ch207": ["numeric", 1],
    #     },  # example; replace with real target(s)
    # },
    # { ... },  # second task example
    # { ... },  # etc.
]
# ---------------------------------------------------------------------------
# ─────────────────── Config helpers ───────────────────────────────────
DEFAULTS = {
    "EARLY_STOP_PATIENCE": 3,
    "MAX_EPOCHS": 2000,
    "LR": 1e-6,
    "BATCH_SIZE": 32,
    "balance_dataset": False,
    "test_only": True,
    #  NEW multi‑GPU / trainer defaults 
    "devices": 4,
    "freeze_positions": False,
    "pooled": True,
    "class-balanced-loss": False,
    "accelerator": "gpu",
    "ddpstrategy": "auto",
    "accumulate_grad_batches": 4,
    "gradient_clip_val": 1,
    "training_precision": "bf16",
    "val_check_interval": 1.0,
    "weight_decay_enc": 1e-2,
    "layer_lr_decay": 0.95,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-6,
    "optimizer_type": "adamw",
    "lr_scheduler": "onecycle",
}

ALWAYS_REQUIRED = [
    "sequence_encoded",
    "result_dir",
    "task_file",
    "model_name",       
    "target_column"
]

REQUIRED_TRAIN = ["train_path", "val_path", "model_save_dir", "pretrained_model_path", "pretrained_model_hparams",]

REQUIRED_TEST =  ["test_path", "checkpoint_path",]


# ---------------------------------------------------------------------------
# Helper functions (from your schema) ----------------------------------------
# ---------------------------------------------------------------------------

def _with_defaults(cfg: dict) -> dict:
    """Merge DEFAULTS and compute derived result_path."""
    out = cfg.copy()
    for k, v in DEFAULTS.items():
        out.setdefault(k, v)
    out["result_path"] = str(Path(out["result_dir"], f"{out['task_file']}.csv"))
    return out


def _integrity_check(cfg: dict):
    """Validate required / forbidden keys according to mode."""
    missing = [k for k in ALWAYS_REQUIRED if k not in cfg]
    if cfg.get("test_only", False):
        missing += [k for k in REQUIRED_TEST if k not in cfg]
        forbidden = [k for k in REQUIRED_TRAIN if k in cfg]
        if forbidden:
            raise ValueError(
                "Config is in test‑only mode but has train/val keys: "
                + ", ".join(forbidden)
            )
    else:
        missing += [k for k in REQUIRED_TRAIN if k not in cfg]
        forbidden = [k for k in REQUIRED_TEST if k in cfg]
        if forbidden:
            raise ValueError(
                "Config is in train mode but has test‑only keys: "
                + ", ".join(forbidden)
            )
    if missing:
        raise ValueError("Missing required keys: " + ", ".join(missing))


def get_best_model_path(val_df, model_name, task_file, target_col, type):
  df = val_df
  m = (df.model_name.eq(model_name) &
       df.task_file.eq(task_file) &
       df.target_col.eq(target_col) &
       df.type.eq(type))
  df = df[m]
  if df.empty: raise ValueError(f'val df is empty for {model_name}, {task_file}, {target_col}, {type}')
  k = 'val_r2' if type == 'numeric' else 'val_auc'
  return df.loc[df[k].idxmax(), 'model_path']

# ---------------------------------------------------------------------------
# Generation logic -----------------------------------------------------------
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 3:
        print(
            "Usage: python gen_config_finetuning_test_only.py <configs_root> <val_results csv>",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg_root = Path(sys.argv[1]).resolve()
    cfg_root.mkdir(parents=True, exist_ok=True)
    val_df = pd.read_csv(sys.argv[2])
    
    registry: list[dict] = []  # rows for CSV/YAML index

    for task in TASKS:
        task_file = task["task_file"]

        # Fill in train/val paths if absent
        task.setdefault("test_path", str(Path(ROOT, "test", f"{task_file}.parquet")))
        for target_col, (target_type, num_outputs) in task["target_column"].items():
            for model_name in MODEL_NAMES:
                cfg = copy.deepcopy(task)
                cfg['target_column'] = {target_col: (target_type, num_outputs)}
                cfg.update(
                    {
                        "sequence_encoded": str(
                            SEQ_OLD if model_name=='medium-random' else SEQ_FULL_POP
                        ),
                        "model_name": model_name,
                        "BATCH_SIZE": BATCH_SIZE,
                        "result_dir": str(
                            Path(RESULT_DIR, task_file, model_name)
                        ),
                        "checkpoint_path": get_best_model_path(
                            val_df, model_name, task_file, target_col, target_type
                        ),
                    }
                )

                # Apply defaults + check integrity
                cfg = _with_defaults(cfg)
                _integrity_check(cfg)

                # Write JSON file
                cfg_dir = cfg_root / task_file / model_name
                cfg_dir.mkdir(parents=True, exist_ok=True)
                cfg_fname = cfg_dir / f"{target_col}_test.json"
                with cfg_fname.open("w") as fp:
                    json.dump(cfg, fp, indent=2)

                # Record in registry
                registry.append(
                    {
                        "task": task_file,
                        "model": model_name,
                        "batch": BATCH_SIZE,
                        "target_column": target_col,
                        "config": str(cfg_fname),
                    }
                )

    # ---------- Write registry artefacts ------------------------------------
    yaml_path = cfg_root / "registry.yaml"
    with yaml_path.open("w") as fp:
        yaml.safe_dump(registry, fp)

    csv_path = cfg_root / "registry.csv"
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, registry[0].keys())
        writer.writeheader()
        writer.writerows(registry)

    print(
        f"Generated {len(registry)} configs under {cfg_root}\n"
        f"Index YAML : {yaml_path}\n"
        f"Index CSV  : {csv_path}"
    )


if __name__ == "__main__":
    main()
