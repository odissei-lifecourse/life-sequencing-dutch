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
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

USER = "ossc"
END_YEAR = 2022
ROW_LIMIT = 0  # 2000
SAMPLE_SIZE = 50000
NA_IDENTIFIER = 9999999999.0


def extract_year(filename):
    matches = re.findall(r"\d{4}", filename)  # Extract all 4-digit numbers
    if len(matches) > 1:
        msg = f"Multiple matches found in filename '{filename}': {matches}"
        raise ValueError(msg)
    elif len(matches) == 0:
        warnings.warn(f"No year found in filename '{filename}'. Ignoring this file")
        return None  # or you can return a default value if you prefer
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
        msg = f"failed to extract some years from {income_files}"
        raise RuntimeError(msg)
    income_data = {}
    for year, file in income_file_dict.items():
        if start_year <= year <= end_year:
            print(f"reading data for {year}")
            file_path = os.path.join(income_dir, file)
            df, meta = pyreadstat.read_sav(file_path, row_limit=ROW_LIMIT, usecols=["RINPERSOON", "INPBELI"])
            df = df.dropna()
            na_mask = df["INPBELI"] == NA_IDENTIFIER
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


def run_cross_validation(df, predictors, target, model, kf, year, train_only_perf=False):
    """Run cross-validation, returning fold results and mean performance."""
    predictors = copy.deepcopy(predictors)
    fold_results = []
    fold = 1

    new_df = df[predictors + [target]].dropna()
    for train_index, test_index in kf.split(new_df):
        train_data, test_data = new_df.iloc[train_index].copy(), new_df.iloc[test_index].copy()

        if "birth_municipality" in predictors or "birth_municipality_encoded" in predictors:
            train_data, test_data = target_encode(train_data, test_data, "birth_municipality", target)
            if "birth_municipality" in predictors:
                predictors.remove("birth_municipality")
            if "birth_municipality_encoded" not in predictors:
                predictors.append("birth_municipality_encoded")

        # Standardize numerical predictors
        numerical_predictors = ["birth_year", "INPBELI_PAST", "birth_municipality_encoded"]
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

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        coefficients = model.coef_
        intercept = model.intercept_

        # Add separate columns for each coefficient
        fold_result = {
            "Fold": fold,
            "MSE": mse,
            "R2": r2,
            "Intercept": intercept,
            "Year": year,
        }
        for predictor, coef in zip(predictors, coefficients):
            fold_result[f"Coeff_{predictor}"] = coef  # Separate columns for each coefficient

        fold_results.append(fold_result)
        fold += 1

    mean_result = {
        "Fold": "mean",
        "MSE": np.mean([result["MSE"] for result in fold_results]),
        "R2": np.mean([result["R2"] for result in fold_results]),
        "Intercept": np.mean([result["Intercept"] for result in fold_results]),
        "Year": year,
    }
    for i, predictor in enumerate(predictors):
        mean_result[f"Coeff_{predictor}"] = np.mean([fold_result[f"Coeff_{predictor}"] for fold_result in fold_results])
    mean_result["dataset_size"] = len(new_df)
    return pd.DataFrame(fold_results + [mean_result])


def custom_format(x):
    f = lambda x: (f"{x:.2e}".replace("+0", "+").replace("-0", "-").replace("+", "")) if abs(x) >= 10000 else f"{x:.2f}"
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
    output_path_formatted = output_path.split(".")[0] + "_formatted." + output_path.split(".")[1]
    formatted_df.to_csv(output_path_formatted, index=False)
    print(f"Results saved to {output_path} and {output_path_formatted}")


def run_primary_experiment(df, output_dir, predictor_year, train_only_perf=False):
    """Run the main experiment with all predictors, saving results for each fold."""
    predictors = ["birth_year", "gender", "birth_municipality", "INPBELI_PAST"]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = LinearRegression()
    all_results = []

    start_year = predictor_year + 1
    for year in range(start_year, END_YEAR + 1):
        target = f"INPBELI_{year}"
        if target not in df.columns:
            warnings.warn(f"Could not find {target} column in dataframe! Must check!")
            continue
        all_results.append(run_cross_validation(df, predictors, target, model, kf, year, train_only_perf))

    final_results_df = pd.concat(all_results, ignore_index=True)
    cols = ["Year", "Fold"] + [col for col in final_results_df.columns if col not in ["Year", "Fold"]]
    final_results_df = final_results_df[cols]

    filename = "primary_experiment_results.csv"
    if train_only_perf:
        filename += "_train_only"
    filename += ".csv"
    save_results_to_csv(output_dir, filename, final_results_df)


def run_additional_experiments(df, output_dir, predictor_year, train_only_perf=False):
    """Run four additional experiments with different subsets of features."""
    experiments = [
        (["INPBELI_PAST"], "Experiment 1: Only INPBELI_PAST"),
        (["INPBELI_PAST", "birth_year"], "Experiment 2: INPBELI_PAST and birth_year"),
        (["INPBELI_PAST", "gender"], "Experiment 3: INPBELI_PAST and gender"),
        (["INPBELI_PAST", "birth_municipality"], "Experiment 4: INPBELI_PAST and birth_municipality"),
        (["birth_year", "gender", "birth_municipality"], "Experiment 5: All excluding INPBELI_PAST"),
        (["gender", "birth_year"], "Experiment 6: birth_year and gender"),
        (["gender"], "Experiment 7: gender"),
        (["birth_year"], "Experiment 8: birth_year'"),
    ]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = LinearRegression()
    all_results = []

    start_year = predictor_year + 1
    for predictors, experiment_name in experiments:
        for year in range(start_year, END_YEAR + 1):
            target = f"INPBELI_{year}"
            if target not in df.columns:
                warnings.warn(f"Could not find {target} column in dataframe! Must check!")
                continue

            fold_results = run_cross_validation(df, predictors, target, model, kf, year, train_only_perf)
            mean_row = fold_results[fold_results["Fold"] == "mean"].iloc[0].to_dict()
            mean_row["Experiment"] = experiment_name
            all_results.append(mean_row)

    final_results_df = pd.DataFrame(all_results)
    cols = ["Experiment", "Year"] + [col for col in final_results_df.columns if col not in ["Experiment", "Year"]]
    final_results_df = final_results_df[cols]

    filename = "additional_experiments_results.csv"
    if train_only_perf:
        filename += "_train_only"
    filename += ".csv"
    save_results_to_csv(output_dir, filename, final_results_df)


def get_mean_gender_income(df, start_year=2017):
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
    df.to_csv("data/temp_output/gender_means.csv")


def main(args):
    """Main function to load data, run experiments, and save results."""
    # Default paths
    data_dir_dict = {"ossc": "/gpfs/ostor/ossc9424/homedir/data/", "snellius": "/projects/0/prjs1019/data/"}
    default_data_dir = data_dir_dict[USER]
    # Command-line arguments
    data_dir = args.data_dir if args.data_dir else default_data_dir
    predictor_year = args.predictor_year

    income_dir = data_dir + "cbs_data/InkomenBestedingen/INPATAB"
    background_path = data_dir + "llm/raw/background.csv"

    output_dir = "data/temp_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load income and background data
    income_data, income_predictor_df = load_income_data(income_dir, predictor_year)
    background_data = load_background_data(background_path, keep_n=SAMPLE_SIZE)
    # Merge past income and background data on RINPERSOON
    merged_df = pd.merge(background_data, income_predictor_df, on="RINPERSOON", how="inner")
    merged_df = merged_df.rename(columns={"INPBELI": "INPBELI_PAST"})
    print(f"merged df size: {len(merged_df)}")

    # Add income data from years after the predictor year
    for year, income_df in income_data.items():
        merged_df = pd.merge(
            merged_df,
            income_df[["RINPERSOON", f"INPBELI_{year}"]],
            on="RINPERSOON",
            how="left",
        )

    get_mean_gender_income(merged_df)
    print("running primary experiment ....")
    run_primary_experiment(merged_df, output_dir, predictor_year, args.train_only)

    print("running additional experiment ....")
    run_additional_experiments(merged_df, output_dir, predictor_year, args.train_only)
    print("all done")


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

    args = parser.parse_args()
    main(args)
