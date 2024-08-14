import pandas as pd
import pyreadstat
import pickle

# Get RINPERSONs of people who we want to check for income
df = pd.read_csv("RINPERSOON_and_income_30.txt",
                delimiter="\t",
                usecols=["RINPERSOON"],
                dtype={'RINPERSOON': int, 'incomeAge30': int, 'birthYear': int})

user_set = set(df['RINPERSOON'])
print(len(user_set))

df = pd.read_spss("INPA2011TABV2.sav",
                usecols=['RINPERSOON', 'INPBELI'])
                #dtype={"RINPERSOON": int, "INPPERSPRIM": int})
                
#df = df.astype({"RINPERSOON": int, "INPPERSPRIM": int})
user_list = list(df['RINPERSOON'])
income_list = list(df['INPBELI'])

baseline_2011 = []

for i in range(len(user_list)):
    try:
        user = int(user_list[i])
        income = income_list[i]
        if income == "9999999999":
            continue
        income = int(income)
    except:
        continue
    
    if user in user_set:
        baseline_2011.append((user, income))
        
print(len(baseline_2011))  
print(baseline_2011[:20])

with open("baseline_2011.pkl", "wb") as pkl_file:
    pickle.dump(baseline_2011, pkl_file)