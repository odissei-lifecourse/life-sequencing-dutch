import os
import sys
import sqlite3
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

# We batch all the inserts and write everything at once at the end of the loop.
records_to_insert = []

for year in marriage_data:

    relevant_marriages = marriage_data[year]

    for person in relevant_marriages:
        partner = relevant_marriages[person]

        real_pair = (person, partner)

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

        records_to_insert.append(record)

# Open a connection to the output database
output_filename = "data/processed/marriage_db.sqlite"
# If output file already exists, then print an error.
if os.path.isfile(output_filename):
    print("Output file already exists: " + output_filename)
    sys.exit(1)

output_conn = sqlite3.connect(output_filename)
output_c = output_conn.cursor()

table_name = "person_marriages"
output_c.execute("""CREATE TABLE """ + table_name +
    """ (rinpersoon NOT NULL PRIMARY KEY, partner NOT NULL, year NOT NULL, fake_partner)""")

# Execute an insert statement with the values for this run.
insert_setup = """INSERT INTO person_income VALUES (?,?,?,?)"""
output_conn.executemany(insert_setup, recordsToInsert)

index_command = ("CREATE INDEX idx_marriage_pair ON person_marriages (rinpersoon, partner)")
output_c.execute(index_command)

# Save (commit) the changes
output_conn.commit()
# Close the output DB
output_conn.close()