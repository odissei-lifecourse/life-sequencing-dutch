#!/usr/bin/env python3
"""
slurm_generator.py
==================

Create one Slurm submission script per configuration JSON listed in the CSV
registry produced by *config_generator.py*.

Usage
-----
    python slurm_generator.py <configs_root> <registry_csv>

     *configs_root*   - the directory that contains the generated JSON
      configuration files (arg0 in your spec).  The scripts will be written to
      'configs_root/slurm_scripts/'.
     *registry_csv*   - the CSV index file created by the config generator
      (arg1).

The generator follows your requirements:
    1. '#SBATCH --job-name' is set to the row **index** of the config (0‑based).
    2. '#SBATCH --nodelist' cycles through the NODES array you provided.
    3. Memory is fixed at **50G**.
    4. 'CUDA_VISIBLE_DEVICES' cycles from 0‑3.
    5. The 'cfg="..."' line points to the config path from the CSV.
    6. All other template lines remain unchanged.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Cluster‑specific settings - tweak if needed
# ---------------------------------------------------------------------------
NODES = ["ossc9424vm1", "ossc9424vm2", "ossc9424vm3"]

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --nodelist={node}
#SBATCH --time=120:00:00
#SBATCH --mem=50G
#SBATCH -p comp_env
#SBATCH -e /gpfs/ostor/ossc9424/logs/%j.%x.err
#SBATCH -o /gpfs/ostor/ossc9424/logs/%j.%x.out

n_gpus=1

echo "job started"
date

source requirements/load_venv.sh
export CUDA_VISIBLE_DEVICES={cuda}

# ---------- run ----------
cfg="{cfg_path}"

date
export NCCL_SOCKET_IFNAME=ib0

srun python -m pop2vec.evaluation.prediction_settings.train_simple "$cfg"

date
echo "job ended successfully"
"""

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage: python slurm_generator.py <configs_root>",
            file=sys.stderr,
        )
        sys.exit(1)

    configs_root = Path(sys.argv[1]).resolve()
    registry_csv = Path(sys.argv[1]).resolve() / 'registry.csv'

    scripts_dir = configs_root / "slurm_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    node_idx, cuda_idx = 0, 0
    with registry_csv.open() as fp:
        reader = csv.DictReader(fp)
        for idx, row in enumerate(reader):
            node = NODES[node_idx]
            cuda = cuda_idx  # 0‑3 cycle
            cfg_path = row["config"]  # path column from registry

            script_content = SLURM_TEMPLATE.format(
                job_name=idx,
                node=node,
                cuda=cuda,
                cfg_path=cfg_path,
            )

            script_file = scripts_dir / f"run_{idx:04d}.sh"
            script_file.write_text(script_content)
            cuda_idx += 1
            if cuda_idx >=4:
                cuda_idx = 0
                node_idx += 1
                if node_idx >= len(NODES):
                    node_idx = 0
    print(f"Generated {idx + 1} Slurm scripts to {scripts_dir}")


if __name__ == "__main__":
    main()
