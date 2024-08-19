import json
import logging
import math
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat
from sklearn.preprocessing import StandardScaler
from .utils import replace_numeric_in_path
from .utils import split_at_last_match
from .utils import split_classes_and_probs
from .utils import transform_dtype


class FakeDataGenerator:
    """Class to generate fake data from summary statistics.

    The class is intended to work with summary statistics that are created with `gen_spreadsheets.py`, and in particular
    requires that each data source is associated with three summary files:
    - *_meta.json: metadata about the data source
    - *_columns.csv
    - *_covariance.csv
    where * is the name of the original data source.

    - For PII variables, it is assumed that each PII identifier is associated with one row.
    - For numerical variables with a low interquartile range (IQR), the unique integers from the range are drawn with equal probability.
    """

    def __init__(self, override=None):
        """Initialize class.

        Args:
            override (dict, optional): if given, overrides the statistics of of the metadata with the provided values.
                This is experimental and used to fix some issues in our input data.
        """
        self.meta = None
        self.col_summary = None
        self.override = override

    def _path_end(self, path_start):
        """From the path to the original file, strip the start string and return the end string."""
        full_path = str(Path(self.meta.get("path")))
        path_start = str(Path(path_start))
        assert path_start in full_path, "invalid start of path provided"
        out = re.sub(path_start, "", full_path)
        if out[0] == "/":
            out = out[1:]
        return out

    def save(self, original_root, new_root, data, replace=None):
        """Save the data to a data root directory with the correct format.

        The new path is defined as follows. From the full original path, `original_root` is cropped and
        replaced with `new_root`. Any remainder of the original path is considered immutable and left unchanged.

        Args:
            original_root (str): original root directory of the source file.
            new_root (str): new root directory of the data.
            data (pd.DataFrame): data frame to store
            replace (dict, optional): value and level to replace in the resulting filename. If provided,
                needs to have keys `"level"` and `"value"`. See the function `replace_numeric_in_path` for details.
        """
        immutable_part = self._path_end(original_root)
        filename = os.path.join(new_root, immutable_part)
        target_dir, _ = os.path.split(filename)
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)

        if replace:
            filename = replace_numeric_in_path(filename, replace["level"], replace["value"])

        if ".csv" in filename:
            data.to_csv(filename, index=False)
        elif ".sav" in filename:
            pyreadstat.write_sav(data, filename)
        else:
            _, filetype = split_at_last_match(filename, r"\.")
            raise NotImplementedError("Cannot save as %s file type" % filetype)

        # XXX deal with different file formats

    def load_metadata(self, url, filename):
        """Load meta data and column summaries.

        Arguments:
            url (str): local path where the summary files are stored
            filename (str): the name of the original data file
        """
        with open(os.path.join(url, filename + "_meta.txt")) as f:
            self.meta = dict(json.load(f))

        # convert "//" in path to "/"
        self.meta["path"] = str(Path(self.meta["path"]))

        self.col_summary = pd.read_csv(os.path.join(url, filename + "_columns.csv"))

    def fit(self):
        """Process metadata and summary statistics to define how each column should be generated."""
        results = {}
        for _, row in self.col_summary.iterrows():
            variable = row["variable_name"]
            results[variable] = detect_variable_type(row)

            if variable in self.override.keys():
                override = self.override[variable]
                if override["type"] == results[variable]["type"]:
                    for feature, value in override["stats"].items():
                        results[variable][feature] = value

        self.generation_inputs = results

    def generate(self, rng, size=None):
        """Generate new data while
            - enforcing non-negativity of numerical variables
            - considering PII columns: each column is an integer range from 0 to `size`.
            - considering null fractions.

        Args:
            rng (np.random.default_rng): random number generator
            size (int, optional): Size of sample to generate. If `None`, the size is taken from
            the data summary.

        Returns:
            pd.DataFrame: the generated data
        """
        assert self.generation_inputs, "You need to `fit` before calling `generate`."
        if size is None:
            size = self.meta.get("total_nobs")
        column_dtypes = self.meta.get("columns_with_dtypes")

        data = {}

        pii_cols = self.meta.get("has_pii_columns")
        for pc in pii_cols:
            x = np.arange(size)
            x = transform_dtype(x, column_dtypes.get(pc))
            data[pc] = x

        for colname, inputs in self.generation_inputs.items():
            required_dtype = column_dtypes.get(colname)

            n_nulls = int(size * inputs["null_fraction"])
            n_nonulls = size - n_nulls

            if inputs["type"] == "categorical":
                col_data = draw_categorical(
                    persons=n_nonulls, values=inputs["classes"], rng=rng, probs=inputs["probs"], return_type="array"
                )
                # col_data = rng.choice(a=inputs["classes"], size=n_nonulls, p=inputs["probs"])
            elif inputs["type"] == "continuous":
                col_data = draw_continuous(rng, inputs["mean"], inputs["std_dev"], n_nonulls, inputs["min"])

            col_data = transform_dtype(col_data, required_dtype)

            if n_nulls > 0:
                col_data = add_nans(rng, col_data, n_nulls)
            data[colname] = col_data

        return pd.DataFrame(data)


def draw_continuous(rng, mean, std_dev, size, min_value):
    """Create a column of continuous data; deal with undesired negative values.

    Args:
        rng (np.random.default_rng): random number generator
        mean, std_dev (float): mean and std for the generated data
        size (int): size of the generated data
        min_value (float): minimum value to be enforced onto the generated data
    """
    col_data = rng.normal(mean, std_dev, size)
    negatives = np.where(col_data < 0)
    col_data[negatives] = min_value
    return col_data


def draw_categorical(persons, values, rng, probs=None, dtype=None, standardize=False, return_type="dict"):
    """Draw categorical values.

    Args:
        persons (np.ndarray or int): If int, the size of the generated samples.
            If np.ndarray, the person identifiers. Determines the number of generated samples.
        values (list or np.ndarray): domain of categorical values.
        rng (np.random.default_rng): random number generator.
        probs (list or np.ndarray, optional): probabilities of classes. If provided, must
            be of same length as `values`. If not provided, all classes are equally likely.
        standardize (bool, optional): if True, values are scaled to a standard normal distribution.
        dtype (str, optional): if provided, the draws are transformed to this data type.
        return_type (str, optional): the type of the return value. Must be either "dict" or "array".
            If "dict", the keys are taken from persons.

    Returns:
        dict[type(persons[0]), np.float64] or np.ndarray

    """
    assert return_type in ["array", "dict"], "please provide a valid return_type"
    assert isinstance(persons, (np.ndarray, int)), "persons needs to be an integer or a np.ndarray"

    if isinstance(persons, np.ndarray):
        nobs = persons.shape[0]
    else:
        nobs = persons

    draws = rng.choice(values, size=nobs, p=probs)

    if dtype:
        draws = transform_dtype(draws, dtype)

    if standardize:
        scaler = StandardScaler()
        draws = draws.reshape(-1, 1)
        draws = scaler.fit_transform(draws)
        draws = draws.squeeze()

    result = draws
    if return_type == "dict":
        if isinstance(persons, int):
            msg = "For creating a dict, provide an array of person identifiers instead of a single integer"
            raise ValueError(msg)
        result = {person: y for person, y in zip(persons, draws)}

    return result


def add_nans(rng, data, size):
    """Add NaNs to an existing array.

    Notes:
        Because numpy cannot store NaNs in a integer array, this changes the types of the output array!
        See also https://stackoverflow.com/questions/12708807/numpy-integer-nan and
        https://numpy.org/doc/stable/glossary.html#term-casting.

    """
    nulls = np.tile(np.nan, size)
    data = np.concatenate([nulls, data])
    rng.shuffle(data)
    return data


def detect_variable_type(row, max_diff_q10_q90=10):
    """Detect how a fake variable should be generated.

    A column is detected as categorical if one of two conditions hold
    - it has at least one category reported
    - it has no category reported, but the summary statistics
      suggest only a limited number of possible classes.

    Because the min is not available in the summaries, the function
    assigns p10 in the actual data to be the min in the data to be generated.

    Args:
        row (dict): a row from a pd.DataFrame
        max_diff_q10_q90 (int, optional): The maximum difference
        between the 10th and 90th percentile in the empirical distribution.
        This is used to infer the second case of categorical variables.

    Returns:
        dict: instructions to generate random variables.

    """
    category_0 = row["category_top_0"]
    if isinstance(category_0, str):
        logging.debug("category_0 is str")
        top_cats = [row[f"category_top_{i}"] for i in range(5)]
        top_cats = [x for x in top_cats if isinstance(x, str)]
        top_cats_splitted = split_classes_and_probs(top_cats)

        classes = [x[0] for x in top_cats_splitted]
        probs = [float(x[1]) for x in top_cats_splitted]
        probsum = sum(probs)
        probs = [prob / probsum for prob in probs]
        result_dict = {"type": "categorical", "classes": classes, "probs": probs}
    else:
        p10, p90 = row["10th_percentile"], row["90th_percentile"]
        if p90 - p10 <= max_diff_q10_q90:
            logging.debug("creating categorical from percentiles")
            pdiff = int(p90) - int(p10)
            # we want to include the end of the range, and deal with cases where p10=p90
            addon = 1 if pdiff > 0 else 2
            p90 += addon
            pdiff += addon
            result_dict = {
                "type": "categorical",
                "classes": np.arange(int(p10), int(p90)),
                "probs": [1 / pdiff for _ in range(pdiff)],
            }
        else:
            result_dict = {"type": "continuous", "mean": row["mean"], "std_dev": row["std_dev"], "min": p10}

    result_dict["null_fraction"] = row["null_fraction"]

    if "probs" in result_dict.keys():
        probs_sum = sum(result_dict["probs"])
        if not math.isclose(1, probs_sum):
            logging.debug("probs do not sum to one!")

    return result_dict
