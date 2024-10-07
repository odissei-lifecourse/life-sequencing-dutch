import pickle
import random

dry_run = True

if dry_run:
    root = "/gpfs/work3/0/prjs1019/data/evaluation/"
else:
    root = "data/processed/"

with open(root + "income_baseline_by_year.pkl", 'rb') as pkl_file:
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

print("Writing income model subset...", flush=True)
with open(root + "income_model_subset.pkl", 'wb') as pkl_file:
    pickle.dump(income_model_set, pkl_file)

print("Writing income eval subset...", flush=True)
with open(root + "income_eval_subset.pkl", 'wb') as pkl_file:
    pickle.dump(income_eval_set, pkl_file)
