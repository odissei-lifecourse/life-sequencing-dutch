import pickle
import random
import sqlite3
import numpy as np

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

records_to_insert = []

for year in marriage_data:

    relevant_marriages = marriage_data[year]

    yearly_pairs = set()

    for person in relevant_marriages:
        partner = relevant_marriages[person]

        partner_gender = gender_map[partner]
        if partner_gender == 1:
            partner_list = full_male_list
        else:
            partner_list = full_female_list

        fake_partner = random.choice(partner_list)

        record = ()
        record += (person,)
        record += (partner,)
        record += (year,)
        record += (fake_partner,)

        # Add in eval flag after we've sampled
        record += (0,)

        records_to_insert.append(record)

        yearly_pairs.add((person, partner))

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

# Open a connection to the output database
output_filename = "data/processed/background_db.sqlite"

sqlite3.register_adapter(np.int64, lambda val: int(val))
sqlite3.register_adapter(np.int32, lambda val: int(val))

output_conn = sqlite3.connect(output_filename)
output_c = output_conn.cursor()

# We batch all the inserts and write everything at once at the end of the loop.
table_name = "person_marriages"
output_c.execute("""CREATE TABLE """ + table_name +
                 """ (rinpersoon NOT NULL, partner NOT NULL, year NOT NULL, fake_partner, is_eval)""")

# Execute an insert statement with the values for this run.
insert_setup = """INSERT INTO person_marriages VALUES (?,?,?,?,?)"""
output_conn.executemany(insert_setup, records_to_insert)

index_command = ("CREATE INDEX idx_marriage_pair ON person_marriages (rinpersoon, partner)")
output_c.execute(index_command)

for year in marriage_data:
    eval_pairs = marriage_eval_sets[year]
    rank_pairs = marriage_rank_sets[year]

    for pair in eval_pairs:
        person = pair[0]
        partner = pair[1]

        update_command = ("UPDATE person_marriages SET is_eval = 1 WHERE rinpersoon = " +
                       str(person) + " AND partner = " + str(partner) + ";")
        output_c.execute(update_command)

# Save (commit) the changes
output_conn.commit()

# Print how many eval sets we inserted
select_command = ('SELECT * FROM person_marriages WHERE is_eval = 1')
results = output_c.execute(select_command)
print('Total Number of Evaluation Pairs:', len(list(results)))

# Save the sets of pairs
with open("data/processed/marriage_model_subset_by_year.pkl", "wb") as pkl_file:
    pickle.dump(marriage_model_sets, pkl_file)

with open("data/processed/marriage_eval_subset_by_year.pkl", "wb") as pkl_file:
    pickle.dump(marriage_eval_sets, pkl_file)

with open("data/processed/marriage_rank_subset_by_year.pkl", "wb") as pkl_file:
    pickle.dump(marriage_rank_sets, pkl_file)