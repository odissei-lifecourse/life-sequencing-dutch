import os
import sys
import sqlite3

input_file = "data/processed/income_by_year.pkl"

with open(input_file, 'rb') as pkl_file:
    data = dict(pickle.load(pkl_file))

# We batch all the inserts and write everything at once at the end of the loop.
records_to_insert = []

for year in data:
    yearly_income = data[year]

    for person in yearly_income:
        income = yearly_income[person]

        record = ()
        record += (person,)
        record += (year,)
        record += (income,)

        records_to_insert.append(record)


# Open a connection to the output database
output_filename = "data/processed/income_db.sqlite"
# If output file already exists, then print an error.
if os.path.isfile(output_filename):
    print("Output file already exists: " + output_filename)
    sys.exit(1)

output_conn = sqlite3.connect(output_filename)
output_c = output_conn.cursor()

table_name = "person_income"
output_c.execute("""CREATE TABLE """ + table_name +
    """ (rinpersoon NOT NULL PRIMARY KEY, year NOT NULL, income NOT NULL)""")

# Execute an insert statement with the values for this run.
insert_setup = """INSERT INTO person_income VALUES (?,?,?)"""
output_conn.executemany(insert_setup, recordsToInsert)

index_command = ("CREATE INDEX idx_income_personyear ON person_income (rinpersoon, year)")
output_c.execute(index_command)

# Save (commit) the changes
output_conn.commit()
# Close the output DB
output_conn.close()