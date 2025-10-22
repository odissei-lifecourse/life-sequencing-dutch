#!/usr/bin/env python3
"""
Combine all CSV files found under the current directory (recursively)
into one file called 'combined.csv'.

All source CSVs must have identical columns.
"""

from pathlib import Path
import pandas as pd
import sys

def main():
    root = Path(sys.argv[1])
    csv_paths = sorted(root.rglob("*.csv"))

    if not csv_paths:
        raise FileNotFoundError("No CSV files were found under this directory.")

    # Read the first file with headers, then read the rest skipping their header rows
    first_df = pd.read_csv(csv_paths[0])
    others   = [pd.read_csv(p) for p in csv_paths[1:]]
    
    combined = pd.concat([first_df, *others], ignore_index=True)

    out_path = root / "combined.csv"
    combined.to_csv(out_path, index=False)
    print(f"\n  Combined {len(csv_paths)} CSV files -> {out_path.resolve()}\n")

if __name__ == "__main__":
    main()