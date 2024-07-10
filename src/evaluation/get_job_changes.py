import pandas as pd
import pickle
import datetime

df = pd.read_csv("/gpfs/ostor/ossc9424/homedir/Life_Course_Evaluation/data/raw/jobs_full_duration.csv", delimiter=';')

job_changes_by_year = {}

for i, row in df.iterrows():
    person_id = int(row['RINPERSOON'])
    
    start_date = datetime.datetime(1971, 12, 30)
    
    days_since_start = int(row['daysSinceFirstEvent'])
    
    end_date = start_date + datetime.timedelta(days=days_since_start)
    
    change_year = end_date.year
    if change_year > 2010:
    
        if change_year not in job_changes_by_year:
            job_changes_by_year[change_year] = set()
            
        job_changes_by_year[change_year].add(person_id)
        
with open("/gpfs/ostor/ossc9424/homedir/Life_Course_Evaluation/data/processed/job_changes_by_year.pkl", 'wb') as pkl_file:
    pickle.dump(job_changes_by_year, pkl_file)