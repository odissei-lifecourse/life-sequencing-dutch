def eval_r2(preds, dtrain):
  labels = dtrain.get_label()
  return 1 - sum((labels - preds) ** 2) / sum((labels - labels.mean()) ** 2)

def get_X_Y(df, label):
  condition = df[label].isna() | np.isinf(df[label])
  df = df[~condition]
  # create X and Y
  X = df.drop(label, axis=1)
  y = df[label]
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
  )
  return X_train, X_test, y_train, y_test

def impute_missing(x):
  x.replace([np.inf, -np.inf], np.nan, inplace=True)
  # Compute the mean of each column, ignoring NaN values
  column_means = x.mean()

  # Fill NaN values with the computed mean of each column
  x.fillna(column_means, inplace=True)
  return x

def get_errors(y_test, pred_test):
  mae = mean_absolute_error(y_test, predictions)
  #print(f'Mean Absolute Error: {mae}')

  rmse = mean_squared_error(y_test, predictions, squared=False)
  #print(f'Root Mean Squared Error: {rmse}')

  r2 = eval_r2(predictions, dtest)
  return mae, rmse, r2


