import pandas as pd
import pickle

df = pd.read_spss("VSLG2023TAB03V1.sav")
print("Loaded data!", flush=True)
df = df[df['gem2010'] == 'Groningen']

building_list = list(df['RINOBJECTNUMMER'])
print(len(building_list))
with open("gron_building_list.pkl", "wb") as pkl_file:
    pickle.dump(building_list, pkl_file)
