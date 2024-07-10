import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support

y_df = pd.read_csv('PreFer_train_outcome.csv')

y_df = y_df[~y_df['new_child'].isna()]
has_outcomes = set(y_df['nomem_encr'].unique())
x_df = pd.read_csv(
  'PreFer_train_background_data.csv', 
  #nrows=1000,
  usecols=['nomem_encr', 'birthyear_imp', 'gender_imp'],
)

x_df = x_df.drop_duplicates(subset='nomem_encr', keep='first')

df = pd.merge(x_df, y_df, on='nomem_encr', how='inner')


print(len(x_df), len(y_df), len(df))

X = df[['birthyear_imp', 'gender_imp']]
X['male'] = X['gender_imp'].apply(lambda x: 1 if x==1 else 0)
X['female'] = X['gender_imp'].apply(lambda x: 0 if x==1 else 1)
#print(X)
X.drop(columns=['gender_imp'], inplace=True)
#print(X)
y = df['new_child'].tolist()

min_max_scaler = MinMaxScaler(feature_range=(0, 1))

# Fit and transform the DataFrame
X = min_max_scaler.fit_transform(X)
#print(scaled_features)
#exit(0)


#clf = LogisticRegression(random_state=0, solver='liblinear')#.fit(X, y)
clf = RandomForestClassifier(random_state=0)

scores = cross_val_score(clf, X, y, cv=10)
print(scores)
print(np.mean(y))
print(X)

clf = LogisticRegression(random_state=0).fit(X, y)

clf = RandomForestClassifier(random_state=0).fit(X, y)
p = clf.predict(X)
#print(p)

# print("-"*100)
# print(clf.classes_)
# print(clf.coef_)
# print(clf.intercept_)
# print(clf.n_features_in_)
# print(clf.feature_names_in_)
# print(clf.n_iter_)

print(precision_recall_fscore_support(y, p, average='weighted'))
