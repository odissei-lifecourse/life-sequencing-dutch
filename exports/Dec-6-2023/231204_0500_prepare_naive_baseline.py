import pandas as pd
import numpy as np
import pickle
import random
from sklearn.preprocessing import OrdinalEncoder


df = pd.read_csv("/gpfs/ostor/ossc9424/homedir/Life_Course_Evaluation/data/raw/new_background.csv", dtype={'RINPERSOON': int, 'birth_year': float, 'gender': int, 'birth_municipality': int})
print(len(df['RINPERSOON'].unique()), flush=True)
df = df.dropna(subset=['RINPERSOON'])
df = df.dropna(subset=['birth_year'])
df = df.dropna(subset=['gender'])
df = df.dropna(subset=['birth_municipality'])

# Only save data for people we might care about
with open("/gpfs/ostor/ossc9424/homedir/Dakota_network/mappings/family_2010.pkl", 'rb') as pkl_file:
    user_set_1 = set(list(dict(pickle.load(pkl_file)).keys()))    
with open("/gpfs/ostor/ossc9424/homedir/Dakota_network/mappings/family_2020.pkl", 'rb') as pkl_file:
    user_set_2 = set(list(dict(pickle.load(pkl_file)).keys()))
    
full_user_set = user_set_1.union(user_set_2)
rando = random.choice(list(full_user_set))
print(rando, flush=True)
print(type(rando), flush=True)

print(len(full_user_set), flush=True)
print("Reducing dataframe", flush=True)

df = df[df['RINPERSOON'].isin(full_user_set)]
print(len(df), flush=True)

person_list = list(df['RINPERSOON'])
birth_year_list = list(df['birth_year'])
gender_list = list(df['gender'])
birth_municipality_list = list(df['birth_municipality'])

print("Normalizing birth years", flush=True)

norm_birth_years = list(np.array(birth_year_list) / np.max(birth_year_list))


print(norm_birth_years[:20], flush=True)

print("Normalizing gender", flush=True)

norm_gender = list(np.array(gender_list) - 1.0)

print(norm_gender[:20], flush=True)


print("Normalizing Municipality...", flush=True)
#enc = OrdinalEncoder(max_categories=10)
#norm_municipality = list(enc.fit_transform(np.array(birth_municipality_list).reshape(-1, 1)))

norm_municipality = list(np.array(birth_municipality_list) / np.max(birth_municipality_list))
print(norm_municipality[:20], flush=True)

birth_year_dict = {}
gender_dict = {}
birth_municipality_dict = {}

for i, person in enumerate(person_list):
    birth_year_dict[person] = norm_birth_years[i]
    gender_dict[person] = norm_gender[i]
    birth_municipality_dict[person] = norm_municipality[i]
        
with open("/gpfs/ostor/ossc9424/homedir/Life_Course_Evaluation/data/processed/person_birth_year.pkl", 'wb') as pkl_file:
    pickle.dump(birth_year_dict, pkl_file)
    
with open("/gpfs/ostor/ossc9424/homedir/Life_Course_Evaluation/data/processed/person_gender.pkl", 'wb') as pkl_file:
    pickle.dump(gender_dict, pkl_file)
    
with open("/gpfs/ostor/ossc9424/homedir/Life_Course_Evaluation/data/processed/person_birth_municipality.pkl", 'wb') as pkl_file:
    pickle.dump(birth_municipality_dict, pkl_file)