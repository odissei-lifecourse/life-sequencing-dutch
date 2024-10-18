import pyreadstat
import pandas as pd
from datetime import datetime
import numpy as np


# Placeholder paths for the input files
data_path = "data/raw_data/"
PARTNERSHIP_FILE_PATH = data_path + "GBAVERBINTENISPARTNER2023BUSV1.sav"
DEATH_FILE_PATH = data_path + "VRLGBAOVERLIJDENTABV2024061.sav"
BIRTH_FILE_PATH = data_path + "VRLGBAPERSOONKTABV2024061.sav"
OUTPUT_FILE_PATH = "data/partnership_after_2016.csv"


PARTNERSHIP_START = 'AANVANGVERBINTENIS'
RINPERSOON = 'RINPERSOON'
PARTNERSHIP_TYPE = 'TYPEVERBINTENIS'


def read_sav_rinpersoon(file_path, columns_to_read=None):
  df, meta = pyreadstat.read_sav(file_path, usecols=columns_to_read)
  df[RINPERSOON] = df[RINPERSOON].astype(int) 
  return df 


def read_partnership_data(file_path):
    """Reads the partnership data and filters the required columns.
    
    Args:
        file_path (str): Path to the partnership data file in .sav format.
        
    Returns:
        pd.DataFrame: A dataframe containing filtered partnership data with valid PARTNERSHIP_TYPE values.
    """
    columns_to_read = [RINPERSOON, PARTNERSHIP_START, PARTNERSHIP_TYPE]
    partnership_data = read_sav_rinpersoon(file_path, columns_to_read)
    # Filter rows with valid PARTNERSHIP_TYPE (only 'H' or 'P')
    partnership_data = partnership_data[
      partnership_data[PARTNERSHIP_TYPE].isin(['H', 'P'])
    ]
    print(f"total # of partnerships over all years = {len(partnership_data)}", flush=True)
    print(f"total # of unique people who got into partnerships= {partnership_data[RINPERSOON].nunique()}", flush=True)
    
    return partnership_data


def read_death_data(file_path):
    """Reads the death data and returns a dataframe with death information.
    
    Args:
        file_path (str): Path to the death data file in .sav format.
        
    Returns:
        pd.DataFrame: A dataframe containing death data.
    """
    columns_to_read = [RINPERSOON, 'VRLGBADatumOverlijden']
    death_data = read_sav_rinpersoon(file_path, columns_to_read)
    print(f"# of deaths over all years = {len(death_data)}", flush=True)  
    return death_data


def read_birth_data(file_path):
    """Reads the birth data and returns a dataframe with birth information.
    
    Args:
        file_path (str): Path to the birth data file in .sav format.
        
    Returns:
        pd.DataFrame: A dataframe containing birth year data.
    """
    columns_to_read = [RINPERSOON, 'VRLGBAGEBOORTEJAAR']
    birth_data = read_sav_rinpersoon(file_path, columns_to_read)
    birth_data['VRLGBAGEBOORTEJAAR'] = birth_data['VRLGBAGEBOORTEJAAR'].astype(
      int
    ) 
    print(f"# of births over all years = {len(birth_data)}",flush=True)
    return birth_data


def filter_first_partnerships(partnership_data, start_year=2017):
    """Filters for people whose first partnership occurred between 2017 and 2023.
    
    Args:
        partnership_data (pd.DataFrame): Dataframe containing partnership information.
        start_year (int): year after which first partnership occurred. 
    Returns:
        pd.DataFrame: Dataframe with people whose first partnership occurred after start_time.
    """
    start_time = f'{start_year}-01-01'
    # Convert PARTNERSHIP_START to datetime
    partnership_data[PARTNERSHIP_START] = pd.to_datetime(
      partnership_data[PARTNERSHIP_START], format='%Y%m%d', errors='coerce'
    )
    
    # Sort by partnership date and drop duplicates to keep the first partnership only
    partnership_data_sorted = partnership_data.sort_values(
      PARTNERSHIP_START
    ).drop_duplicates(subset=[RINPERSOON], keep='first')
    
    # Filter for partnerships between 2017 and 2023
    partnership_data_filtered = partnership_data_sorted[
        (partnership_data_sorted[PARTNERSHIP_START] >= start_time)
    ]

    for year in range(2017, 2025):
      print(f"# of first partnerships in {year} = {partnership_data_sorted[PARTNERSHIP_START].astype(str).str.startswith(str(year)).sum()}", flush=True)
    
    # Add f'first_union_after_{start_year-1}' column
    partnership_data_filtered[f'first_union_after_{start_year-1}'] = 1
    
    return partnership_data_filtered[[RINPERSOON, f'first_union_after_{start_year-1}']]


def filter_eligible_non_married(
  birth_data, 
  death_data, 
  partnership_data, 
  start_year=2017
):
    """Filters people who are alive, not older than 80, and have never been married.
    
    Args:
        birth_data (pd.DataFrame): Dataframe containing birth year information.
        death_data (pd.DataFrame): Dataframe containing death information.
        partnership_data (pd.DataFrame): Dataframe containing partnership information.
        start_year (int): year after which first partnership occurred. 
    Returns:
        pd.DataFrame: Dataframe with people who meet the criteria.
    """
    # Calculate age and filter for people 80 years old or younger
    birth_data['age'] = start_year - birth_data['VRLGBAGEBOORTEJAAR']
    eligible_birth_data = birth_data[
      (birth_data['age'] >= 18) & (birth_data['age'] <= 80)
    ]

    print(f"# of dead+alive people between 18-80 = {len(eligible_birth_data)}", flush=True)    
    # Merge with death data to exclude deceased individuals
    alive_data = pd.merge(
      eligible_birth_data, 
      death_data, 
      on=RINPERSOON, 
      how='left', 
      indicator=True
    )
    alive_data = alive_data[alive_data['_merge'] == 'left_only']  # Keep only those not in the death data
    alive_data = alive_data.drop(columns=['_merge'])

    print(f"# of alive people = {len(alive_data)}", flush=True)    

    # Exclude people who got married
    unique_partnership_data = partnership_data[['RINPERSOON']].drop_duplicates()
    print(f"# of people who ever got into partnership = {len(unique_partnership_data)}", flush=True)
    eligible_non_married = pd.merge(
      alive_data, 
      unique_partnership_data, 
      on=RINPERSOON, 
      how='left', 
      indicator=True
    )
    eligible_non_married = eligible_non_married[
      eligible_non_married['_merge'] == 'left_only'
    ]  # Keep only those not in the partnership data
    
    print(f"# of people with label = 0 {len(eligible_non_married)})", flush=True)  
    # Add f'first_union_after_{start_year-1}' column with 0 for non-married
    eligible_non_married[f'first_union_after_{start_year-1}'] = 0
    
    return eligible_non_married[[RINPERSOON, f'first_union_after_{start_year-1}']]


def main():
    start_year = 2017
    # Read the input files
    partnership_data = read_partnership_data(PARTNERSHIP_FILE_PATH)
    death_data = read_death_data(DEATH_FILE_PATH)
    birth_data = read_birth_data(BIRTH_FILE_PATH)
    
    # Filter data for first partnerships after start_year
    first_partnerships = filter_first_partnerships(partnership_data, start_year)
    
    # Filter data for people who never got married, are alive, and not older than 80
    eligible_non_married = filter_eligible_non_married(
      birth_data, 
      death_data, 
      partnership_data,
      start_year
    )
    
    overlap = set(first_partnerships[RINPERSOON]) & set(eligible_non_married[RINPERSOON])
    assert len(overlap) == 0, (
      f"Overlap set size between first_partnerships and eligible_non_married "
      f"is {len(overlap)}.\n {'-'*20} The overlapping set is {overlap}"
    )

    # Combine the data and save to CSV
    combined_data = pd.concat(
      [first_partnerships, eligible_non_married], 
      ignore_index=True
    )
    assert combined_data[RINPERSOON].is_unique, f'final dataframe does not have unique values for {RINPERSOON}'

    label = f'first_union_after_{start_year-1}'
    print(
      f"""
      final dataset length = {len(combined_data)},
      # of positives = {np.sum(combined_data[label]==1)}
      # of negatives = {np.sum(combined_data[label]==0)}
      """
    )
    combined_data.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"Data has been processed and saved to {OUTPUT_FILE_PATH}")


if __name__ == "__main__":
    main()
