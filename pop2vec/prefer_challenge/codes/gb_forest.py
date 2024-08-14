import pandas as pd
import pyreadstat
import numpy as np

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

from sklearn.linear_model import LinearRegression

class GB: 
  def impute_missing(x):
    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Compute the mean of each column, ignoring NaN values
    column_means = x.mean()

    # Fill NaN values with the computed mean of each column
    x.fillna(column_means, inplace=True)
    return x

  def baseline(X_train, X_test, y_train):

    X_train_new = impute_missing(copy.deepcopy(X_train))
    X_test_new = copy.deepcopy(X_test)

    reg = LinearRegression().fit(X_train_new, y_train)
    preds = reg.predict(X_test_new)
    return preds

  def get_errors(y_test, pred_test):
    mae = mean_absolute_error(y_test, predictions)
    #print(f'Mean Absolute Error: {mae}')

    rmse = mean_squared_error(y_test, predictions, squared=False)
    #print(f'Root Mean Squared Error: {rmse}')

    r2 = eval_r2(predictions, dtest)
    return mae, rmse, r2

  def train_and_eval_for_label(df, label, optimal_num_trees=-1):
    X, y = get_X_Y(df, label)
    # Split the data into train and test sets (80:20 split)
    X_train, X_test, y_train, y_test = get_X_Y(df, label)
    baseline_predictions = baseline(X_train, X_test, y_train)
    # Define XGBoost parameters for a regression model
    xgb_params = {
        'objective': 'reg:squarederror',
        'booster': 'gbtree',
        'max_depth': 6,  # Can be tuned via cross-validation
        'min_child_weight': 1,  # Can be tuned via cross-validation
        'eta': 0.1,  # Start with 0.1, can be lowered if necessary
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }

    # Convert your dataset into DMatrix, which is optimized for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.nan)
    dtest = xgb.DMatrix(X_test, label=y_test)

    if optimal_num_trees == -1:
      # Set the number of boosting rounds
      num_boost_round = 200  # Can be tuned via cross-validation

      # Train the model using cross-validation to find the optimal number of trees
      cv_results = xgb.cv(xgb_params, 
        dtrain, 
        num_boost_round=num_boost_round, 
        nfold=5,
        metrics='rmse', 
        early_stopping_rounds=10, 
        seed=42, 
        #feval=eval_r2
      )
      #print(cv_results)
      # Train the model with the optimal number of trees
      optimal_num_trees = cv_results.shape[0]
      print(f"optimal # of trees = {optimal_num_trees}")
    
    xgb_model = xgb.train(
      params=xgb_params, 
      dtrain=dtrain, 
      num_boost_round=optimal_num_trees
    )

    # Predict on test set
    predictions = xgb_model.predict(dtest)
    baseline_predictions = baseline(X_train, X_test, y_train)
    # Evaluate predictions using RMSE
    #print("# of test instances", len(y_test))
    #print(f'R^2 Error: {r2}')
    return len(y_test), np.mean(y_test), np.std(y_test), mae, rmse, r2 

  def __init__(self, file_path, columns_to_drop=None,):
  # Read the .sav file
    if file_path.endswith('.sav'):
      df, meta = pyreadstat.read_sav(file_path)
    elif file_path.endswith('.csv'):
      df = pd.read_csv(file_path)
    else:
      raise ValueError("file format not supported, must be sav or csv")
    #columns_to_drop = ['nomem_encr', 'nohouse_encr', 'cv10c_m']  # Example list of columns to drop

    # Drop columns from 'df' that are in 'columns_to_drop'
    if columns_to_drop not None:
      df = df.drop([col for col in columns_to_drop if col in df.columns], axis=1)



# Path to your .sav file
fp = 'data/private/LISS/ODISSEI Summer School 2023 - Gert Stulp/Politics and Values/source/cv10c_EN_1.0p.sav'
df, meta = load_data(fp)
#label = 'cv10c001'
labels = []
test_ns = []
means = []
stds = []
maes = []
rmses = []
r2s = []
questions = []
options = []
for label in tqdm(df.columns):
  #print(label)
  labels.append(label)
  question = meta.column_names_to_labels.get(label, "not found")
  option = meta.variable_value_labels.get(label, "not found")
  questions.append(question)
  options.append(option)

  test_n, mean, std, mae, rmse, r2 = train_and_eval_for_label(df, label, 60)
  test_ns.append(test_n)
  means.append(mean)
  stds.append(std)
  maes.append(mae) 
  rmses.append(rmse)
  r2s.append(r2)

result_df = pd.DataFrame(
  {
    'label':labels,
    'question': questions, 
    'option': options,
    'test_n': test_ns,
    'mean': means,
    'std_dev': stds,
    'mae': maes,
    'rmse': rmses,
    'r2': r2s
  }
)

result_df.to_csv('code/experimental/LISS/POLITICS_AND_VALUES_2010_results.csv')