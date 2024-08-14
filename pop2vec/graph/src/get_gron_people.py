import pandas as pd
import pickle

print("Loaded buildings", flush=True)
with open("gron_building_list.pkl", "rb") as pkl_file:
    building_set = set(pickle.load(pkl_file))
    
df = pd.read_spss("GBAADRESOBJECT2022BUSV1.sav")
print("Loaded data!", len(df), flush=True)
df = df[df['RINOBJECTNUMMER'].isin(building_set)]
print("Parsed down to only Groningen residents:", len(df), flush=True)

#years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020] 
years = [2009]
for year in years:

    residents = []
    for i, row in df.iterrows():
        first_year = int(row['GBADATUMAANVANGADRESHOUDING'][:4])
        last_year = int(row['GBADATUMEINDEADRESHOUDING'][:4])
        
        if year >= first_year and year <= last_year:
            rin = int(row['RINPERSOON'])
            residents.append(rin)
           
    print("Saving", len(residents), "residents", flush=True)
    with open("gron_" + str(year) + "_resident_list.pkl", "wb") as pkl_file:
        pickle.dump(residents, pkl_file)