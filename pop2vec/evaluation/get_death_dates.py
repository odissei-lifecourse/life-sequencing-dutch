import pandas as pd
import pickle
import datetime

df = pd.read_spss("/gpfs/ostor/ossc9424/homedir/Life_Course_Evaluation/data/raw/VRLGBAOVERLIJDENTABV2023092.sav")
print("Loaded data!", flush=True)

person_death_by_year = {}

for i, row in df.iterrows():
    person_id = int(row['RINPERSOON'])
    death_date_str = row['VRLGBADatumOverlijden']
    
    year = int(death_date_str[:4])
    if year > 2010:
    
        person_death_by_year[person_id] = year
        
with open("/gpfs/ostor/ossc9424/homedir/Life_Course_Evaluation/data/processed/death_years_by_person.pkl", 'wb') as pkl_file:
    pickle.dump(person_death_by_year, pkl_file)

