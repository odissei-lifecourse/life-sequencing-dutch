import os
import sys
import sqlite3


# 1. Birth Year (Age)
with open("/gpfs/ostor/ossc9424/homedir/Life_Course_Evaluation/data/processed/person_birth_year.pkl", 'rb') as pkl_file:
    person_birth_year = dict(pickle.load(pkl_file))

# 2. Gender
with open("/gpfs/ostor/ossc9424/homedir/Life_Course_Evaluation/data/processed/person_gender.pkl", 'rb') as pkl_file:
    person_gender = dict(pickle.load(pkl_file))

# 3. Birth City
with open("/gpfs/ostor/ossc9424/homedir/Life_Course_Evaluation/data/processed/person_birth_municipality.pkl", 'rb') as pkl_file:
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

    records_to_insert.append(record)


# Open a connection to the output database
output_filename = "data/processed/background_db.sqlite"
# If output file already exists, then print an error.
if os.path.isfile(output_filename):
    print("Output file already exists: " + output_filename)
    sys.exit(1)

output_conn = sqlite3.connect(output_filename)
output_c = output_conn.cursor()

table_name = "person_background"
output_c.execute("""CREATE TABLE """ + table_name +
    """ (rinpersoon NOT NULL PRIMARY KEY, gender NOT NULL, birth_year NOT NULL, birth_city NOT NULL)""")

# Execute an insert statement with the values for this run.
insert_setup = """INSERT INTO person_income VALUES (?,?,?,?)"""
output_conn.executemany(insert_setup, recordsToInsert)

index_command = ("CREATE INDEX idx_background_person ON person_background (rinpersoon)")
output_c.execute(index_command)

# Save (commit) the changes
output_conn.commit()
# Close the output DB
output_conn.close()