import pickle
import random

with open("data/processed/income_baseline_by_year.pkl", 'rb') as pkl_file:
    data = dict(pickle.load(pkl_file))

income_model_set = set()
income_eval_set = set()

person_set = set()

for year in data:

    yearly_income = data[year]
    for person in yearly_income:
        person_set.add(person)

isolated_group = list(random.sample(list(person_set), 60000))

income_model_set = set(isolated_group[:50000])
income_eval_set = set(isolated_group[50000:])

with open("data/processed/income_model_subset.pkl", 'wb') as pkl_file:
    pickle.dump(income_model_set, pkl_file)

with open("data/processed/income_eval_subset.pkl", 'wb') as pkl_file:
    pickle.dump(income_eval_set, pkl_file)
