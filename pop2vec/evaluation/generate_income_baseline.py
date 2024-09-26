import os
import pickle
import re
import numpy as np
import pandas as pd
import pyreadstat
from tqdm import tqdm

source_dir = "/gpfs/ostor/ossc9424/homedir/cbs_data/real/InkomenBestedingen/INPATAB/"
target_dir = "data/processed/"

baseline_by_years = {}
years = [x for x in range(2011, 2023)]

inpa_files = os.listdir(source_dir)

for f in tqdm(inpa_files):
    filename = os.path.join(source_dir, f)
    df, _ = pyreadstat.read_sav(filename, usecols=["RINPERSOON", "INPBELI"])

    # make sure we record the year correctly and only have 1 file per year
    year_matches = re.findall(r"\d{4}", f)
    assert len(year_matches) == 1
    year = int(year_matches[0])
    years = [y for y in years if y != year]

    user_list = list(df["RINPERSOON"])
    income_list = list(df["INPBELI"])

    yearly_baseline = {}
    num_zeros = 0
    num_unfound = 0

    for i in range(len(user_list)):
        try:
            user = int(user_list[i])
            income = income_list[i]
            if income == "9999999999":
                num_unfound += 1
                continue

            income = int(income)
        except:
            continue

        if income == 0:
            num_zeros += 1

        yearly_baseline[user] = income

    # Print some summary statistics to verify that our data looks similar for every year
    income_values = list(yearly_baseline.values())

    print("---------------------------------------", flush=True)
    print("Summary of year:", str(year), flush=True)
    print("Number of records: ", str(len(income_values)), flush=True)
    print("Maximum value: ", str(max(income_values)), flush=True)
    print("Minimum value: ", str(min(income_values)), flush=True)
    print("Average value: ", str(np.mean(income_values)), flush=True)
    print("Number of Zeros: ", str(num_zeros), flush=True)
    print("Number of Unfound: ", str(num_unfound), flush=True)
    print("Dtype of User List: ", str(df.dtypes["RINPERSOON"]))
    print("Dtype of Income: ", str(df.dtypes["INPBELI"]))

    baseline_by_years[year] = yearly_baseline

with open(os.path.join(target_dir, "income_baseline_by_year.pkl"), "wb") as pkl_file:
    pickle.dump(baseline_by_years, pkl_file)
