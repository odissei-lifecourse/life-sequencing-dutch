import sys
import os
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pyreadstat
import copy


END_YEAR = 2021
ROW_LIMIT = 0 # 2000
def load_income_data(income_dir, predictor_year):
    """Load income data from multiple SAV files."""
    # Load all income files starting from predictor_year+1 to the last available year
    start_year = predictor_year + 1
    end_year = END_YEAR
    income_files = [f'INPA{year}TABV4.sav' for year in range(start_year, end_year+1)]
    
    income_data = {}
    for file in income_files:
        year = int(file[4:8])  # Extract year from file name, e.g., '2017' from 'INPA2017TABV4.sav'
        if start_year <= year <= end_year:
          print(f"reading data for {year}")
          file_path = os.path.join(income_dir, file)
          df, meta = pyreadstat.read_sav(
            file_path, row_limit=ROW_LIMIT, usecols=['RINPERSOON', 'INPBELI']
          )
          df = df.dropna()
          income_data[year] = df.rename(columns={'INPBELI': f'INPBELI_{year}'})

    # Load the predictor year income file (used as predictor)
    predictor_file = os.path.join(income_dir, f'INPA{predictor_year}TABV4.sav')
    income_predictor_df, _ = pyreadstat.read_sav(
      predictor_file, row_limit=ROW_LIMIT, usecols=['RINPERSOON', 'INPBELI']
    )
    income_predictor_df = income_predictor_df.dropna()
    return income_data, income_predictor_df

def load_background_data(background_path, keep_n=None):
    """Load background data from CSV."""
    df = pd.read_csv(background_path, dtype={'RINPERSOON': 'object'})
    df = df.rename(
      columns={
        'year': 'birth_year', 
        'municipality': 'birth_municipality'
      }
    )
    df = df.dropna()
    if keep_n and keep_n < len(df):
      df = df.sample(n=keep_n, random_state=42)
    return df

def target_encode(train_data, test_data, target_column, target):
    """Encode the birth_municipality with mean target values from training data."""
    train_means = train_data.groupby(target_column)[target].mean()
    train_data[f'{target_column}_encoded'] = train_data[target_column].map(train_means)
    test_data[f'{target_column}_encoded'] = test_data[target_column].map(train_means)
    global_mean = train_data[target].mean()
    test_data[f'{target_column}_encoded'].fillna(global_mean, inplace=True)

    return train_data, test_data

def normalize_data(train_data, test_data, predictors):
    """Normalize the numerical predictors in the training and testing sets."""
    scaler = StandardScaler()
    train_data[predictors] = scaler.fit_transform(train_data[predictors])
    test_data[predictors] = scaler.transform(test_data[predictors])
    
    return train_data, test_data

def run_cross_validation(df, predictors, target, model, kf):
    """Run cross-validation, returning fold results and mean performance."""
    predictors = copy.deepcopy(predictors)
    fold_results = []
    fold = 1
    # Check which cells contain NaN
    nan_locations = df.isnull()

    # Display rows with NaN values
    rows_with_nan = df[nan_locations.any(axis=1)]
    # print("*"*100)
    # print(rows_with_nan)
    # print("*"*100)
    new_df = df[predictors + [target]].dropna()
    for train_index, test_index in kf.split(new_df):
        train_data, test_data = new_df.iloc[train_index].copy(), new_df.iloc[test_index].copy()

        if 'birth_municipality' in predictors or 'birth_municipality_encoded' in predictors:
            train_data, test_data = target_encode(train_data, test_data, 'birth_municipality', target)
            if 'birth_municipality' in predictors:
              predictors.remove('birth_municipality')
            if 'birth_municipality_encoded' not in predictors:
              predictors.append('birth_municipality_encoded')

        # Standardize numerical predictors
        numerical_predictors = ['birth_year', 'INPBELI_PAST', 'birth_municipality_encoded']
        train_data, test_data = normalize_data(train_data, test_data, [p for p in numerical_predictors if p in predictors])

        train_data = train_data[predictors + [target]].dropna()

        # Drop rows with NaN in any of the predictor or target columns in test_data
        test_data = test_data[predictors + [target]].dropna()

        X_train = train_data[predictors]
        y_train = train_data[target]
        X_test = test_data[predictors]
        y_test = test_data[target]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        coefficients = model.coef_
        intercept = model.intercept_

        fold_results.append({
            "Fold": fold,
            "MSE": mse,
            "R2": r2,
            "Intercept": intercept,
            "Coefficients": coefficients
        })
        fold += 1

    mean_mse = np.mean([result['MSE'] for result in fold_results])
    mean_r2 = np.mean([result['R2'] for result in fold_results])
    mean_coefficients = np.mean([result['Coefficients'] for result in fold_results], axis=0)
    mean_intercept = np.mean([result['Intercept'] for result in fold_results])

    return fold_results, mean_mse, mean_r2, mean_coefficients, mean_intercept, len(new_df)

def save_results_to_csv(output_dir, filename, results_df):
    """Save the results DataFrame to CSV."""
    output_path = os.path.join(output_dir, filename)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

def run_primary_experiment(df, output_dir, predictor_year):
    """Run the main experiment with all predictors, saving results for each fold."""
    predictors = ['birth_year', 'gender', 'birth_municipality', 'INPBELI_PAST']
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = LinearRegression()
    all_results = []

    start_year = predictor_year + 1
    # print(range(start_year, END_YEAR+1))
    # print(df.columns)
    for year in range(start_year, END_YEAR+1):
        target = f'INPBELI_{year}'
        if target not in df.columns:
            continue


        fold_results, mean_mse, mean_r2, mean_coefficients, mean_intercept, _ = run_cross_validation(
            df, predictors, target, model, kf
        )

        fold_results_df = pd.DataFrame(fold_results)
        fold_results_df['Year'] = year

        # Append mean result
        mean_row = pd.DataFrame({
            "Fold": ["Mean"],
            "MSE": [mean_mse],
            "R2": [mean_r2],
            "Intercept": [mean_intercept],
            "Coefficients": [mean_coefficients],
            "Year": [year]
        })
        fold_results_df = pd.concat([fold_results_df, mean_row], ignore_index=True)

        all_results.append(fold_results_df)

    final_results_df = pd.concat(all_results, ignore_index=True)
    save_results_to_csv(output_dir, "primary_experiment_results.csv", final_results_df)

def run_additional_experiments(df, output_dir, predictor_year):
    """Run four additional experiments with different subsets of features."""
    experiments = [
        (['INPBELI_PAST'], 'Experiment 1: Only INPBELI_PAST'),
        (['INPBELI_PAST', 'birth_year'], 'Experiment 2: INPBELI_PAST and birth_year'),
        (['INPBELI_PAST', 'gender'], 'Experiment 3: INPBELI_PAST and gender'),
        (['INPBELI_PAST', 'birth_municipality'], 'Experiment 4: INPBELI_PAST and birth_municipality')
    ]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = LinearRegression()
    all_results = []

    start_year = predictor_year + 1
    for predictors, experiment_name in experiments:
        for year in range(start_year, END_YEAR+1):
            target = f'INPBELI_{year}'
            if target not in df.columns:
                continue

            _, mean_mse, mean_r2, mean_coefficients, mean_intercept, dataset_size = run_cross_validation(
                df, predictors, target, model, kf
            )

            # Save only the mean result for each year
            all_results.append({
                "Experiment": experiment_name,
                "Year": year,
                "MSE": mean_mse,
                "R2": mean_r2,
                "Intercept": mean_intercept,
                "Coefficients": mean_coefficients,
                "dataset_size": dataset_size,
            })

    final_results_df = pd.DataFrame(all_results)
    save_results_to_csv(output_dir, "additional_experiments_results.csv", final_results_df)

def main():
    """Main function to load data, run experiments, and save results."""
    # Default paths
    default_data_dir = '/projects/0/prjs1019/data/'
    # Command-line arguments
    data_dir = sys.argv[1] if len(sys.argv) > 1 else default_data_dir
    predictor_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2016
    
    income_dir = data_dir + 'cbs_data/InkomenBestedingen/INPATAB'
    background_path = data_dir + 'llm/raw/data/background.csv'
    
    
    output_dir = 'data/temp_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load income and background data
    income_data, income_predictor_df = load_income_data(income_dir, predictor_year)
    background_data = load_background_data(background_path, keep_n=50000)
    # Merge past income and background data on RINPERSOON
    merged_df = pd.merge(
      background_data, income_predictor_df, on='RINPERSOON', how='inner'
    )
    merged_df = merged_df.rename(columns={'INPBELI':'INPBELI_PAST'})
    print(f"merged df size: {len(merged_df)}")
    
    # Add income data from years after the predictor year
    for year, income_df in income_data.items():
        merged_df = pd.merge(
          merged_df, 
          income_df[['RINPERSOON', f'INPBELI_{year}']], 
          on='RINPERSOON', 
          how='left',
        )

    print("running primary experiment ....")
    run_primary_experiment(merged_df, output_dir, predictor_year)

    print("running additional experiment ....")
    run_additional_experiments(merged_df, output_dir, predictor_year)
    print("all done")


if __name__ == '__main__':
  main()
