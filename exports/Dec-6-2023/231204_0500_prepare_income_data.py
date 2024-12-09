import numpy as np
import pandas as pd
import pickle
import random

# Get RINPERSONs of people who we want to check for income
df = pd.read_csv("data/raw/RINPERSOON_and_income_30.txt",
                delimiter="\t",
                dtype={'RINPERSOON': int, 'incomeAge30': int, 'birthYear': int})
                
print("Found income data for:", len(df), "people", flush=True)


full_grouped_by_year = {}

for i, row in df.iterrows():
    user = row['RINPERSOON']
    income = row['incomeAge30']
    year = row['birthYear']

    year += 30
    
    if year not in full_grouped_by_year:
        full_grouped_by_year[year] = {}
    
    full_grouped_by_year[year][user] = income
    
with open("data/processed/income_by_year.pkl", "wb") as pkl_file:
    pickle.dump(full_grouped_by_year, pkl_file)