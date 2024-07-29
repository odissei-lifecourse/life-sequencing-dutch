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

marriage_model_sets = {}
marriage_eval_sets = {}
marriage_rank_sets = {}

for year in marriage_data:

    relevant_marriages = marriage_data[year]

    yearly_pairs = set()

    for person in relevant_marriages:
        partner = relevant_marriages[person]

        real_pair = (person, partner)

        partner_gender = gender_map[partner]
        if partner_gender == 1:
            partner_list = full_male_list
        else:
            partner_list = full_female_list

        fake_partner = random.choice(partner_list)
        fake_pair = (person, fake_partner)

        yearly_pairs.append(real_pair)
        yearly_pairs.append(fake_pair)

    # Grab 5000 people from each year
    isolated_group = list(random.sample(list(yearly_pairs), 5000))

    # 4000 for model training
    model_group = isolated_group[:4000]
    # 1000 for eval
    eval_group = isolated_group[4000:]

    # Add the people to the corresponding sets
    marriage_model_sets[year] = set(model_group)
    marriage_eval_sets[year] = set(eval_group)
    marriage_rank_sets[year] = set(random.sample(model_group, 1000))

# Save the sets of pairs
with open("data/processed/marriage_model_subset_by_year.pkl", "wb") as pkl_file:
    pickle.dump(marriage_model_sets, pkl_file)

with open("data/processed/marriage_eval_subset_by_year.pkl", "wb") as pkl_file:
    pickle.dump(marriage_eval_sets, pkl_file)

with open("data/processed/marriage_rank_subset_by_year.pkl", "wb") as pkl_file:
    pickle.dump(marriage_rank_set, pkl_file)
