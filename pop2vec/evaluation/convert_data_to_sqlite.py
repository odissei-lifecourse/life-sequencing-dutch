import os
import sys
import sqlite3
import pickle
import random
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dry_run",
    default=False,
    type=bool,
)

args = parser.parse_args()
dry_run = args.dry_run

if dry_run:
    root = "/gpfs/work3/0/prjs1019/data/evaluation/"
else:
    root = "/gpfs/ostor/ossc9424/homedir/Life_Course_Evaluation/data/processed/"

# Open a connection to the output database
output_filename = root + "background_db.sqlite"

sqlite3.register_adapter(np.int64, lambda val: int(val))
sqlite3.register_adapter(np.int32, lambda val: int(val))

# If output file already exists, then print an error.
# if os.path.isfile(output_filename):
#     print("Output file already exists: " + output_filename)
#     sys.exit(1)

output_conn = sqlite3.connect(output_filename)
output_c = output_conn.cursor()

########################################################################################################################

records_to_insert = []
input_file = root + "income_baseline_by_year.pkl"

with open(root + "income_eval_subset.pkl", 'rb') as pkl_file:
    income_eval_set = set(pickle.load(pkl_file))

with open(input_file, 'rb') as pkl_file:
    data = dict(pickle.load(pkl_file))

for year in data:
    yearly_income = data[year]

    for person in yearly_income:
        income = yearly_income[person]

        record = ()
        record += (person,)
        record += (year,)
        record += (income,)

        if person in income_eval_set:
            record += (1,)
        else:
            record += (0,)

        records_to_insert.append(record)

table_name = "person_income"
output_c.execute("""CREATE TABLE """ + table_name +
    """ (rinpersoon NOT NULL
        , year NOT NULL
        , income NOT NULL
        , is_eval
        , PRIMARY KEY ( rinpersoon, year )
        )""")

# Execute an insert statement with the values for this run.
insert_setup = """INSERT INTO person_income VALUES (?,?,?,?)"""
output_conn.executemany(insert_setup, records_to_insert)

index_command = ("CREATE INDEX idx_income_personyear ON person_income (rinpersoon, year)")
output_c.execute(index_command)

# Save (commit) the changes
output_conn.commit()

########################################################################################################################

with open(root + "marriages_by_year.pkl", "rb") as pkl_file:
    marriage_data = dict(pickle.load(pkl_file))

with open(root + "id_to_gender_map.pkl", "rb") as pkl_file:
    gender_map = dict(pickle.load(pkl_file))

with open(root + "full_male_list.pkl", "rb") as pkl_file:
    full_male_list = list(pickle.load(pkl_file))

with open(root + "full_female_list.pkl", "rb") as pkl_file:
    full_female_list = list(pickle.load(pkl_file))

with open(root + "marriage_eval_subset.pkl", "rb") as pkl_file:
    marriage_eval_set = set(pickle.load(pkl_file))

with open(root + "marriage_rank_subset.pkl", "rb") as pkl_file:
    marriage_rank_set = set(pickle.load(pkl_file))

# We batch all the inserts and write everything at once at the end of the loop.
records_to_insert = []

for year in marriage_data:

    relevant_marriages = marriage_data[year]

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

        if person in marriage_eval_set or person in marriage_rank_set:
            record += (1,)
        elif partner in marriage_eval_set or partner in marriage_rank_set:
            record += (1,)
        else:
            record += (0,)

        records_to_insert.append(record)

table_name = "person_marriages"
output_c.execute("""CREATE TABLE """ + table_name +
    """ (rinpersoon NOT NULL, partner NOT NULL, year NOT NULL, fake_partner, is_eval)""")

# Execute an insert statement with the values for this run.
insert_setup = """INSERT INTO person_marriages VALUES (?,?,?,?,?)"""
output_conn.executemany(insert_setup, records_to_insert)

index_command = ("CREATE INDEX idx_marriage_pair ON person_marriages (rinpersoon, partner)")
output_c.execute(index_command)

# Save (commit) the changes
output_conn.commit()

########################################################################################################################

full_set = income_eval_set 

# 1. Birth Year (Age)
with open(root + "person_birth_year.pkl", 'rb') as pkl_file:
    person_birth_year = dict(pickle.load(pkl_file))

# 2. Gender
with open(root + "person_gender.pkl", 'rb') as pkl_file:
    person_gender = dict(pickle.load(pkl_file))

# 3. Birth City
with open(root + "person_birth_municipality.pkl", 'rb') as pkl_file:
    person_birth_city = dict(pickle.load(pkl_file))

# We batch all the inserts and write everything at once at the end of the loop.
records_to_insert = []

for person in person_birth_year:
    birth_year = person_birth_year[person]
    gender = person_gender[person]
    birth_city = person_birth_city[person]

    record = ()
    record += (person,)
    record += (gender,)
    record += (birth_year,)
    record += (birth_city,)

    if person in full_set:
        record += (1,)
    else:
        record += (0,)

    records_to_insert.append(record)

table_name = "person_background"
output_c.execute("""CREATE TABLE """ + table_name +
    """ (rinpersoon NOT NULL PRIMARY KEY, gender NOT NULL, birth_year NOT NULL, birth_city NOT NULL, is_eval)""")

# Execute an insert statement with the values for this run.
insert_setup = """INSERT INTO person_background VALUES (?,?,?,?,?)"""
output_conn.executemany(insert_setup, records_to_insert)

index_command = ("CREATE INDEX idx_background_person ON person_background (rinpersoon)")
output_c.execute(index_command)

# Save (commit) the changes
output_conn.commit()

########################################################################################################################

# Close the output DB
output_conn.close()
