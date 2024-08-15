import pandas as pd
import pickle
from tqdm import tqdm
import os 
import re 

source_dir = "/gpfs/ostor/ossc9424/homedir/cbs_data/real/InkomenBestedingen/INPATAB/"
target_dir = "data/processed/"

baseline_by_years = {}
years = [x for x in range(2011, 2023)]

inpa_files = os.listdir(source_dir)

for f in tqdm(inpa_files):
    filename = os.path.join(source_dir, f)
    df = pd.read_spss(filename,
                      usecols=['RINPERSOON', 'INPBELI'])
    
    # make sure we record the year correctly and only have 1 file per year
    year_matches = re.findall(r"\d{4}", f)
    assert len(year_matches) == 1
    year = int(year_matches[0])
    years = [y for y in years if y != year]

    user_list = list(df['RINPERSOON'])
    income_list = list(df['INPBELI'])

    yearly_baseline = {}

    for i in range(len(user_list)):
        try:
            user = int(user_list[i])
            income = income_list[i]
            if income == "9999999999":
                continue
            income = int(income)
        except:
            continue

        yearly_baseline[user] = income

    baseline_by_years[year] = yearly_baseline


with open(os.path.join(target_dir, "income_baseline_by_year.pkl"), "wb") as pkl_file:
    pickle.dump(baseline_by_years, pkl_file)