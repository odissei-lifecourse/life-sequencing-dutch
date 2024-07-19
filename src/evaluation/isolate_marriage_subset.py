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

marriage_model_set = set()
marriage_eval_set = set()

for year in marriage_data:

    relevant_marriages = marriage_data[year]

    yearly_people = set()

    for person in relevant_marriages:
        partner = relevant_marriages[person]

        # Only 1 real marriage per person per year
        if person in yearly_people or partner in yearly_people:
            continue

        yearly_people.add(person)
        yearly_people.add(partner)

        #seen_people.add(person)
        #seen_people.add(partner)

        real_pair = (person, partner)

        partner_gender = gender_map[partner]

        if partner_gender == 1:
            partner_list = full_male_list
        else:
            partner_list = full_female_list

        fake_partner = random.choice(partner_list)
        yearly_people.add(fake_partner)

    # Grab 5000 people from each year
    isolated_group = list(random.sample(yearly_people, 5000))

    # 4000 for model training
    model_group = isolated_group[:4000]
    # 1000 for eval
    eval_group = isolated_group[4000:]

    # Add the people to the corresponding sets
    marriage_model_set.update(model_group)
    marriage_eval_set.update(eval_group)

# Grab 1000 people from the training group for rank prediction
marriage_rank_set = set(random.sample(marriage_model_set, 1000))

# Save the sets of IDs
with open("data/processed/marriage_model_subset.pkl", "wb") as pkl_file:
    pickle.dump(marriage_model_set, pkl_file)

with open("data/processed/marriage_eval_subset.pkl", "wb") as pkl_file:
    pickle.dump(marriage_eval_set, pkl_file)

with open("data/processed/marriage_rank_subset.pkl", "wb") as pkl_file:
    pickle.dump(marriage_rank_set, pkl_file)
