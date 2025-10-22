import json
import re
from pathlib import Path
import pandas as pd


def f(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """
    Read a JSON file consisting of a list of objects like:
        {
          "input": "RINPERSOON: 123456, ........., RINPERSOON: 2993 .......",
          "output": "0"
        }
    Extract the *first two* occurrences of the pattern 'RINPERSOON: {digits}' (note the colon)
    from the 'input' string and produce a DataFrame with columns:
        - RINPERSOON1
        - RINPERSOON2
        - is_twin   (int 0/1 from the original "output" field)

    The DataFrame is written to `output_path` as a Parquet file (no index) and also returned.

    Parameters
    ----------
    input_path : str | Path
        Path to the source JSON file.
    output_path : str | Path
        Path to the parquet file to create (directories will be created if needed).

    Returns
    -------
    pd.DataFrame
        The resulting DataFrame.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compile regex once: match 'RINPERSOON:' with optional whitespace then digits
    pattern = re.compile(r'RINPERSOON:\s*(\d+)')

    with input_path.open('r', encoding='utf-8') as f_in:
        data = json.load(f_in)

    rows = []
    for idx, record in enumerate(data):
        text = record.get("input", "")
        outputs = record.get("output", "")

        matches = pattern.findall(text)

        if len(matches) != 2:
            raise ValueError(
                f"Entry {idx} does not contain exactly two 'RINPERSOON:' occurrences with digits. "
                f"Found {len(matches)} in text: {text!r}"
            )

        rin1, rin2 = matches[0], matches[1]

        try:
            is_twin = int(outputs)
        except (TypeError, ValueError):
            raise ValueError(f"Entry {idx} has non-integer 'output' value: {outputs!r}")
        assert is_twin in [0, 1]
        rows.append(
            {
                "RINPERSOON1": int(rin1),
                "RINPERSOON2": int(rin2),
                "is_twin": is_twin
            }
        )

    df = pd.DataFrame(rows, columns=["RINPERSOON1", "RINPERSOON2", "is_twin"])

    df["RINPERSOON1"] = df["RINPERSOON1"]
    df["RINPERSOON2"] = df["RINPERSOON2"]
    df["is_twin"] = df["is_twin"].astype(int)

    # Write parquet (requires pyarrow or fastparquet)
    df.to_parquet(output_path, index=False)

    return df


# Example usage:
# df_result = f("input.json", "output.parquet")
