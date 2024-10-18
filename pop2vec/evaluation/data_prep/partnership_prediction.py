import pyreadstat
import pandas as pd
from datetime import datetime
import numpy as np


# Placeholder paths for the input files
PARTNERSHIP_FILE_PATH = "path_to_partnership_data.sav"
DEATH_FILE_PATH = "path_to_death_data.sav"
BIRTH_FILE_PATH = "path_to_birth_data.sav"
OUTPUT_FILE_PATH = "path_to_output.csv"


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
    
    # Add f'first_union_after_{start_year}' column
    partnership_data_filtered[f'first_union_after_{start_year}'] = 1
    
    return partnership_data_filtered[[RINPERSOON, f'first_union_after_{start_year}']]


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
    eligible_birth_data = birth_data[birth_data['age'] <= 80]
    
    # Merge with death data to exclude deceased individuals
    alive_data = pd.merge(
      eligible_birth_data, 
      death_data, 
      on=RINPERSOON, 
      how='left', 
      indicator=True
    )
    alive_data = alive_data[alive_data['_merge'] == 'left_only']  # Keep only those not in the death data
    

    # Exclude people who got married
    unique_partnership_data = partnership_data[['RINPERSOON']].drop_duplicates()
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
    
    # Add f'first_union_after_{start_year}' column with 0 for non-married
    eligible_non_married[f'first_union_after_{start_year}'] = 0
    
    return eligible_non_married[[RINPERSOON, f'first_union_after_{start_year}']]


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

    label = f'first_union_after_{start_year}'
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
