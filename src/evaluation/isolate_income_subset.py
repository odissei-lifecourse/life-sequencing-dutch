import pickle

with open("data/processed/income_by_year.pkl", 'rb') as pkl_file:
    data = dict(pickle.load(pkl_file))

person_set = set()

for year in data:
    yearly_income = data[year]
    for person in yearly_income:
        person_set.add(person)

with open("data/processed/income_subset.pkl", 'wb') as pkl_file:
    pickle.dump(person_set, pkl_file)