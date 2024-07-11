import pickle
import random

with open("data/processed/marriages_by_year.pkl", "rb") as pkl_file:
    marriage_data = dict(pickle.load(pkl_file))

with open("data/processed/id_to_gender_map.pkl", "rb") as pkl_file:
    gender_map = dict(pickle.load(pkl_file))

with open("data/processed/full_male_list.pkl", "rb") as pkl_file:
    full_male_list = list(pickle.load(pkl_file))

with open("data/processed/full_female_list.pkl", "rb") as pkl_file:
    full_female_list = list(pickle.load(pkl_file))

seen_people = set()

for year in marriage_data:

    relevant_marriages = marriage_data[year]

    for person in relevant_marriages:
        partner = relevant_marriages[person]

        if person in seen_people or partner in seen_people:
            continue

        seen_people.add(person)
        seen_people.add(partner)

        real_pair = (person, partner)

        partner_gender = gender_map[partner]

        if partner_gender == 1:
            partner_list = full_male_list
        else:
            partner_list = full_female_list

        fake_partner = random.choice(partner_list)
        seen_people.add(fake_partner)

with open("data/processed/marriage_subset.pkl", "wb") as pkl_file:
    pickle.dump(seen_people, pkl_file)
