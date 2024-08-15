import numpy as np
import pandas as pd
import pickle
import random

yearly_marriage_data = {}
yearly_partner_data = {}
gender_map = {}

# Get RINPERSONs of people who we want to check for marriages
df = pd.read_csv("data/raw/unions.csv",
                 delimiter=';')
print(len(df))
df = df.dropna()  
print(len(df))              
                 
df = df.astype({'RINPERSOON': int, 'RINPart2': int, 'dateBegRelation': str,
                'typeRel': str, 'genderPart1': int, 'genderPart2': int})

                
male_df = df[df['genderPart1'] == 1]
full_male_list = list(male_df['RINPERSOON'])

female_df = df[df['genderPart1'] == 2]
full_female_list = list(female_df['RINPERSOON'])

num_partners = 0
num_marriages = 0

for i, row in df.iterrows():
    person_1 = row['RINPERSOON']
    person_2 = row['RINPart2']
    
    relation_date = row['dateBegRelation']
    relation_type = row['typeRel']
    
    gender_1 = row['genderPart1']
    gender_2 = row['genderPart2']
    
    year = int(relation_date[:4])
    
    if relation_type == 'P':
    # Add the partnership into the partner data for this year
        if year not in yearly_partner_data:
            yearly_partner_data[year] = {}
            
        num_partners += 1
        
        if person_1 not in yearly_partner_data[year]:
            yearly_partner_data[year][person_1] = person_2
        if person_2 not in yearly_partner_data[year]:
            yearly_partner_data[year][person_2] = person_1
                
    elif relation_type == 'H':
        if year not in yearly_marriage_data:
            yearly_marriage_data[year] = {}
            
        num_marriages += 1
        
        if person_1 not in yearly_marriage_data[year]:
            yearly_marriage_data[year][person_1] = person_2
        if person_2 not in yearly_marriage_data[year]:
            yearly_marriage_data[year][person_2] = person_1

    # Add these people to the gender map
    if person_1 not in gender_map:
        gender_map[person_1] = gender_1
    if person_2 not in gender_map:
        gender_map[person_2] = gender_2

    
with open("data/processed/id_to_gender_map.pkl", "wb") as pkl_file:
    pickle.dump(gender_map, pkl_file)
    
with open("data/processed/partnerships_by_year.pkl", "wb") as pkl_file:
    pickle.dump(yearly_partner_data, pkl_file)
    
with open("data/processed/marriages_by_year.pkl", "wb") as pkl_file:
    pickle.dump(yearly_marriage_data, pkl_file)
    
with open("data/processed/full_male_list.pkl", "wb") as pkl_file:
    pickle.dump(full_male_list,  pkl_file)
    
with open("data/processed/full_female_list.pkl", "wb") as pkl_file:
    pickle.dump(full_female_list, pkl_file)
    
print("Num Partnerships: " + str(num_partners))
print("Num Marriages: " + str(num_marriages))
