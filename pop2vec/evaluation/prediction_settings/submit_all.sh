#!/usr/bin/env bash
#
# submit_all.sh
#
# Usage:
#   ./submit_all.sh /path/to/slurm/scripts
#   # If no path is given, the current directory is used.

set -euo pipefail

# Directory that holds the Slurm scripts
DIR="${1:-.}"

# Find every *.sh file (non‑recursively) and submit it
for f in "$DIR"/*.sh; do
    # Skip the loop if no files match
    [[ -e "$f" ]] || { echo "No .sh files found in $DIR"; break; }

    echo "Submitting $f"
    sbatch "$f"
done
