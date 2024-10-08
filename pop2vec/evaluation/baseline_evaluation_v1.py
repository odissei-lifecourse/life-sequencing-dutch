import argparse
import copy
import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
import pyreadstat
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from pop2vec.evaluation.baseline_v1_utils import (
    target_transformation,
    TRANSFORMATION_FUNCTIONS,
)
from pop2vec.evaluation.report_utils import load_hdf5

USER = "snellius"  # "ossc"
END_YEAR = 2022
ROW_LIMIT = 2000
SAMPLE_SIZE = 50000
NA_IDENTIFIERS = [9999999999.0, 0]


def extract_year(filename):
    matches = re.findall(r"\d{4}", filename)  # Extract all 4-digit numbers
    if len(matches) > 1:
        msg = f"Multiple matches found in filename '{filename}': {matches}"
        raise ValueError(msg)
    elif len(matches) == 0:
        warnings.warn(f"No year found in filename '{filename}'. Ignoring this file")
        return None
    else:
        return int(matches[0])


def load_income_data(income_dir, predictor_year):
    """Load income data from multiple SAV files."""
    # Load all income files starting from predictor_year+1 to the last available year
    start_year = predictor_year
    end_year = END_YEAR
    income_files = os.listdir(income_dir)
    income_file_dict = {extract_year(file): file for file in income_files}
    if len(income_file_dict) != len(income_files):
        msg = f"Failed to extract some years from {income_files}"
        raise RuntimeError(msg)
    income_data = {}
    for year, file in income_file_dict.items():
        if start_year <= year <= end_year:
            print(f"Reading data for {year}")
            file_path = os.path.join(income_dir, file)
            df, meta = pyreadstat.read_sav(
                file_path, row_limit=ROW_LIMIT, usecols=["RINPERSOON", "INPBELI"]
            )
            df = df.dropna()
            na_mask = df["INPBELI"].isin(NA_IDENTIFIERS)
            df = df.loc[~na_mask, :]
            income_data[year] = df.rename(columns={"INPBELI": f"INPBELI_{year}"})

    # Load the predictor year income file (used as predictor)
    income_predictor_df = income_data.pop(predictor_year)
    return income_data, income_predictor_df


def load_background_data(background_path, keep_n=None):
    """Load background data from CSV."""
    df = pd.read_csv(background_path, dtype={"RINPERSOON": "object"})
    df = df.rename(columns={"year": "birth_year", "municipality": "birth_municipality"})
    df = df.dropna()
    if keep_n and keep_n < len(df):
        df = df.sample(n=keep_n)  # , random_state=42)
    df["gender"] -= 1
    return df


def get_embedding_df(embeddings_path, embedding_type):
    """Load embeddings from HDF5 file."""
    global USER
    person_key = 'sequence_id'
    rinpersoon_ids, embeddings = load_hdf5(
        emb_url=embeddings_path,
        id_key=person_key,
        value_key=embedding_type,
        sample_size=ROW_LIMIT if ROW_LIMIT != 0 else -1,
    )
    if USER == 'snellius':  # TODO: this needs to be fixed in a future commit
        rinpersoon_ids = np.array([str(i) for i in range(len(embeddings))])
    # Create a DataFrame for embeddings
    embedding_dim = embeddings.shape[1]
    embedding_columns = [f"embedding_{i}" for i in range(embedding_dim)]
    embeddings_df = pd.DataFrame(embeddings, columns=embedding_columns)
    embeddings_df["RINPERSOON"] = rinpersoon_ids

    return embeddings_df


def target_encode(train_data, test_data, target_column, target):
    """Encode the birth_municipality with mean target values from training data."""
    train_means = train_data.groupby(target_column)[target].mean()
    train_data[f"{target_column}_encoded"] = train_data[target_column].map(train_means)
    test_data[f"{target_column}_encoded"] = test_data[target_column].map(train_means)
    global_mean = train_data[target].mean()
    test_data[f"{target_column}_encoded"].fillna(global_mean, inplace=True)

    return train_data, test_data


def normalize_data(train_data, test_data, numeric_predictors):
    """Normalize the numerical predictors in the training and testing sets."""
    scaler = StandardScaler()
    if len(numeric_predictors) > 0:
        train_data[numeric_predictors] = scaler.fit_transform(train_data[numeric_predictors])
        test_data[numeric_predictors] = scaler.transform(test_data[numeric_predictors])

    return train_data, test_data


def run_cross_validation(
    df,
    predictors,
    target,
    model,
    kf,
    year,
    train_only_perf=False,
    transformation=None,
    embedding_columns=None,
):
    """Run cross-validation, returning fold results and mean performance."""
    if embedding_columns is None:
        embedding_columns = []
    predictors = copy.deepcopy(predictors)
    fold_results = []
    fold = 1

    # Replace 'embedding' with actual embedding columns
    if 'embedding' in predictors:
        predictors = [col if col != 'embedding' else embedding_columns for col in predictors]
        # Flatten the list if embedding_columns is a list of lists
        predictors = [item for sublist in predictors for item in (sublist if isinstance(sublist, list) else [sublist])]

    new_df = df[predictors + [target]].dropna()
    if transformation:
        new_df[target], transformation_info = target_transformation(
            new_df[target].to_numpy(),
            transformation,
        )
    for train_index, test_index in kf.split(new_df):
        train_data, test_data = new_df.iloc[train_index].copy(), new_df.iloc[test_index].copy()

        if "birth_municipality" in predictors or "birth_municipality_encoded" in predictors:
            train_data, test_data = target_encode(train_data, test_data, "birth_municipality", target)
            if "birth_municipality" in predictors:
                predictors.remove("birth_municipality")
            if "birth_municipality_encoded" not in predictors:
                predictors.append("birth_municipality_encoded")

        # Standardize numerical predictors
        numerical_predictors = ["birth_year", "INPBELI_PAST", "birth_municipality_encoded"] + [
            col for col in embedding_columns if col in predictors
        ]
        train_data, test_data = normalize_data(
            train_data, test_data, [p for p in numerical_predictors if p in predictors]
        )

        train_data = train_data[predictors + [target]].dropna()

        # Drop rows with NaN in any of the predictor or target columns in test_data
        test_data = test_data[predictors + [target]].dropna()

        X_train = train_data[predictors]
        y_train = train_data[target]
        X_test = test_data[predictors]
        y_test = test_data[target]

        model.fit(X_train, y_train)
        if train_only_perf:
            X_test, y_test = X_train, y_train

        y_pred = model.predict(X_test)
        if transformation:
            y_pred = target_transformation(
                target_array=y_pred,
                transformation_type=transformation,
                inverse=True,
                transformation_info=transformation_info,
            )
            y_test = target_transformation(
                target_array=y_test.to_numpy(),
                transformation_type=transformation,
                inverse=True,
                transformation_info=transformation_info,
            )
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        coefficients = model.coef_
        intercept = model.intercept_

        # Add separate columns for each coefficient
        fold_result = {
            "Fold": fold,
            "MSE": mse,
            "R2": r2,
            "MAPE": mape,
            "Intercept": intercept,
            "Year": year,
        }
        embedding_coefficients = []
        for predictor, coef in zip(predictors, coefficients):
            if predictor in embedding_columns:
                embedding_coefficients.append(coef)
            else:
                fold_result[f"Coeff_{predictor}"] = coef  # Separate columns for non-embedding coefficients

        # Sum absolute values of embedding coefficients
        if embedding_coefficients:
            embedding_coeff_sum = np.sum(np.abs(embedding_coefficients))
            fold_result["Coeff_embeddings"] = embedding_coeff_sum

        fold_results.append(fold_result)
        fold += 1

    # Compute mean results
    mean_result = {
        "Fold": "mean",
        "MSE": np.mean([result["MSE"] for result in fold_results]),
        "R2": np.mean([result["R2"] for result in fold_results]),
        "MAPE": np.mean([result["MAPE"] for result in fold_results]),
        "Intercept": np.mean([result["Intercept"] for result in fold_results]),
        "Year": year,
    }
    # For non-embedding predictors
    for predictor in set(predictors) - set(embedding_columns):
        mean_result[f"Coeff_{predictor}"] = np.mean(
            [fold_result.get(f"Coeff_{predictor}", 0) for fold_result in fold_results]
        )
    # For embeddings, average the sum of absolute values
    if embedding_columns and any("Coeff_embeddings" in fr for fr in fold_results):
        mean_result["Coeff_embeddings"] = np.mean(
            [fold_result.get("Coeff_embeddings", 0) for fold_result in fold_results]
        )
    mean_result["dataset_size"] = len(new_df)
    return pd.DataFrame(fold_results + [mean_result])


def custom_format(x):
    f = (
        lambda x: (f"{x:.2e}".replace("+0", "+").replace("-0", "-").replace("+", ""))
        if abs(x) >= 10000
        else f"{x:.2f}"
    )
    if isinstance(x, str):
        return x
    elif isinstance(x, (int, float)):
        return f(x)
    else:
        warnings.warn(f"Found non-string non-numeric. x= {x}, type = {type(x)}")
        return x


def save_results_to_csv(output_dir, filename, results_df):
    """Save the results DataFrame to CSV."""
    output_path = os.path.join(output_dir, filename)
    results_df.to_csv(output_path, index=False)

    # Apply custom formatting to the entire DataFrame
    formatted_df = results_df.applymap(custom_format)
    output_path_formatted = (
        output_path.split(".")[0] + "_formatted." + output_path.split(".")[1]
    )
    formatted_df.to_csv(output_path_formatted, index=False)
    print(f"Results saved to {output_path} and {output_path_formatted}")


def run_primary_experiment(
    df, output_dir, predictor_year, train_only_perf=False, transformation=None, use_embeddings=False
):
    """Run the main experiment with all predictors, saving results for each fold."""
    embedding_columns = [col for col in df.columns if col.startswith("embedding_")] if use_embeddings else []
    predictors = ["birth_year", "gender", "birth_municipality", "INPBELI_PAST"] + embedding_columns
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = LinearRegression()
    all_results = []

    start_year = predictor_year + 1
    for year in range(start_year, END_YEAR + 1):
        target = f"INPBELI_{year}"
        if target not in df.columns:
            warnings.warn(f"Could not find {target} column in dataframe! Must check!")
            continue
        all_results.append(
            run_cross_validation(
                df,
                predictors,
                target,
                model,
                kf,
                year,
                train_only_perf,
                transformation,
                embedding_columns,
            )
        )

    final_results_df = pd.concat(all_results, ignore_index=True)
    cols = ["Year", "Fold"] + [col for col in final_results_df.columns if col not in ["Year", "Fold"]]
    final_results_df = final_results_df[cols]

    filename = f"primary_results_{transformation}"
    if train_only_perf:
        filename += "_train_only"
    filename += ".csv"
    save_results_to_csv(output_dir, filename, final_results_df)


def run_additional_experiments(
    df, output_dir, predictor_year, train_only_perf=False, transformation=None, use_embeddings=False
):
    """Run additional experiments with different subsets of features."""
    embedding_columns = [col for col in df.columns if col.startswith("embedding_")] if use_embeddings else []

    # Base experiments without embeddings
    experiments = [
        (["INPBELI_PAST"], "Experiment 1: Only INPBELI_PAST"),
        (["INPBELI_PAST", "birth_year"], "Experiment 2: INPBELI_PAST and birth_year"),
        (["INPBELI_PAST", "gender"], "Experiment 3: INPBELI_PAST and gender"),
        (["INPBELI_PAST", "birth_municipality"], "Experiment 4: INPBELI_PAST and birth_municipality"),
        (["birth_year", "gender", "birth_municipality"], "Experiment 5: All excluding INPBELI_PAST"),
        (["gender", "birth_year"], "Experiment 6: birth_year and gender"),
        (["gender"], "Experiment 7: gender"),
        (["birth_year"], "Experiment 8: birth_year"),
    ]

    # Include additional experiments only when embeddings are used
    if use_embeddings:
        additional_experiments = [
            (["embedding"], "Experiment 9: Embeddings only"),
            (["embedding", "birth_year"], "Experiment 10: Embeddings and birth_year"),
            (["embedding", "gender"], "Experiment 11: Embeddings and gender"),
            (["embedding", "birth_municipality"], "Experiment 12: Embeddings and birth_municipality"),
            (["embedding", "INPBELI_PAST"], "Experiment 13: Embeddings and INPBELI_PAST"),
        ]
        experiments.extend(additional_experiments)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = LinearRegression()
    all_results = []

    start_year = predictor_year + 1
    for predictors, experiment_name in experiments:
        # Replace 'embedding' with actual embedding columns
        if 'embedding' in predictors:
            predictors = [col if col != 'embedding' else embedding_columns for col in predictors]
            # Flatten the list if necessary
            predictors = [item for sublist in predictors for item in (sublist if isinstance(sublist, list) else [sublist])]

        for year in range(start_year, END_YEAR + 1):
            target = f"INPBELI_{year}"
            if target not in df.columns:
                warnings.warn(f"Could not find {target} column in dataframe! Must check!")
                continue

            fold_results = run_cross_validation(
                df,
                predictors,
                target,
                model,
                kf,
                year,
                train_only_perf,
                transformation,
                embedding_columns if use_embeddings else [],
            )
            mean_row = fold_results[fold_results["Fold"] == "mean"].iloc[0].to_dict()
            mean_row["Experiment"] = experiment_name
            all_results.append(mean_row)

    final_results_df = pd.DataFrame(all_results)
    cols = ["Experiment", "Year"] + [
        col for col in final_results_df.columns if col not in ["Experiment", "Year"]
    ]
    final_results_df = final_results_df[cols]

    filename = f"additional_results_{transformation}"
    if train_only_perf:
        filename += "_train_only"
    filename += ".csv"
    save_results_to_csv(output_dir, filename, final_results_df)


def get_mean_gender_income(df, output_dir, start_year=2017):
    results = []
    for year in range(start_year, END_YEAR + 1):
        male_mean = np.nanmean(df[df["gender"] == 0][f"INPBELI_{year}"])
        female_mean = np.nanmean(df[df["gender"] == 1][f"INPBELI_{year}"])
        results.append(
            {
                "year": year,
                "mean": np.nanmean(df[f"INPBELI_{year}"]),
                "male_mean": male_mean,
                "female_mean": female_mean,
                "m-f": male_mean - female_mean,
                "male_count": len(df[df["gender"] == 0]),
                "female_count": len(df[df["gender"] == 1]),
            }
        )
    df = pd.DataFrame(results)
    df = df.applymap(custom_format)
    df.to_csv(os.path.join(output_dir, "gender_means.csv"))


def main(args):
    """Main function to load data, run experiments, and save results."""
    # Default paths
    data_dir_dict = {
        "ossc": "/gpfs/ostor/ossc9424/homedir/data/",
        "snellius": "/projects/0/prjs1019/data/",
    }
    default_data_dir = data_dir_dict[USER]
    # Command-line arguments
    data_dir = args.data_dir if args.data_dir else default_data_dir
    predictor_year = args.predictor_year

    income_dir = data_dir + "cbs_data/InkomenBestedingen/INPATAB"
    background_path = data_dir + "llm/raw/background.csv"

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load income and background data
    income_data, income_predictor_df = load_income_data(income_dir, predictor_year)
    background_data = load_background_data(background_path, keep_n=SAMPLE_SIZE)
    # Merge past income and background data on RINPERSOON
    merged_df = pd.merge(background_data, income_predictor_df, on="RINPERSOON", how="inner")
    merged_df = merged_df.rename(columns={f"INPBELI_{predictor_year}": "INPBELI_PAST"})
    print(f"Merged df size: {len(merged_df)}")

    # Add income data from years after the predictor year
    for year, income_df in income_data.items():
        merged_df = pd.merge(
            merged_df,
            income_df[["RINPERSOON", f"INPBELI_{year}"]],
            on="RINPERSOON",
            how="left",
        )

    use_embeddings = args.embeddings_path is not None

    if use_embeddings:
        # Load embeddings
        embeddings_path = args.embeddings_path
        embedding_type = args.embedding_type
        embeddings_df = get_embedding_df(embeddings_path, embedding_type)

        # Merge embeddings with merged_df
        merged_df = pd.merge(merged_df, embeddings_df, on="RINPERSOON", how="inner")
        print(f"Merged (After embedding loading) df size: {len(merged_df)}")

    get_mean_gender_income(merged_df, output_dir)
    for f in TRANSFORMATION_FUNCTIONS:
        print(f"Transformation_function = {f}\nRunning primary experiment ....")
        run_primary_experiment(
            merged_df, output_dir, predictor_year, args.train_only, f, use_embeddings=use_embeddings
        )

        print(f"Transformation_function = {f}\nRunning additional experiment ....")
        run_additional_experiments(
            merged_df, output_dir, predictor_year, args.train_only, f, use_embeddings=use_embeddings
        )
    print("All done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-only",
        dest="train_only",
        action=argparse.BooleanOptionalAction,
        help="If given, results refer only to training set",
    )
    parser.add_argument(
        "--predictor-year",
        dest="predictor_year",
        type=int,
        default=2016,
        help="Year from which the income feature to take",
    )
    parser.add_argument(
        "--data-dir",
        dest="data_dir",
        type=str,
        default=None,
        help="Base data directory",
    )
    parser.add_argument(
        "--embeddings-path",
        dest="embeddings_path",
        type=str,
        default=None,
        help="Path to the embeddings HDF5 file. If not provided, embeddings will not be used.",
    )
    parser.add_argument(
        "--embedding-type",
        dest="embedding_type",
        type=str,
        default="embeddings",
        help="Type of the embedding to be used, one of [cls_emb, mean_emb, embeddings]",
    )

    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        default="data/temp_output",
        help="The output directory",
    )

    args = parser.parse_args()
    main(args)
