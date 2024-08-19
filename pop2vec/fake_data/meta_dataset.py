"""Class holding metadata summary from a file."""

from __future__ import annotations
import json
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path


@dataclass
class MetaDataSet:
    """Class holding file metadata."""
    path: str
    shape: list[int]
    columns_with_dtypes: dict[str, str]
    total_nobs: int
    nobs_sumstat: dict[str, int] | None = None
    has_pii_columns: list[str] = field(default_factory=list)

    def __post_init__(self):
        if len(self.shape) != 2: #noqa: PLR2004
            msg = "Shape must be a list with exactly two elements"
            raise ValueError(msg)

        if len(self.columns_with_dtypes) != self.shape[1]:
            msg = "Number of columns in columns_with_dtypes must match second element of shape"
            raise ValueError(msg)

    @classmethod
    def from_json(cls, json_path: str) -> MetaDataSet:
        """Read JSON file and create a DatasetInfo instance.

        Args:
            json_path (str): Path to the JSON file.

        Returns:
            MetaDataSet: An instance of DatasetInfo populated with data from the JSON file.
        """
        with Path(json_path).open("r") as f:
            data = json.load(f)

        # Ensure all required fields are present
        required_fields = ["path", "shape", "columns_with_dtypes", "total_nobs", "has_pii_columns"]
        for f in required_fields:
            if f not in data:
                msg = f"Required field '{f}' is missing in the JSON file."
                raise ValueError(msg)

        # Create the instance
        return cls(
            path=str(Path(data["path"])), # convert "//" to "/"
            shape=data["shape"],
            columns_with_dtypes=data["columns_with_dtypes"],
            total_nobs=data["total_nobs"],
            nobs_sumstat=data.get("nobs_sumstat"),
            has_pii_columns=data.get("has_pii_columns", [])
        )



