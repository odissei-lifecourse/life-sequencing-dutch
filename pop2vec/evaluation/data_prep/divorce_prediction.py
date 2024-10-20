import pandas as pd
import pyreadstat

# Constants for column names
PERSON_ID = "RINPERSOON"
PARTNER_ID = "RINPERSOONVERBINTENISP"
START_DATE = "AANVANGVERBINTENIS"
END_DATE = "EINDEVERBINTENIS"
DISSOLUTION_REASON = "REDENBEEINDIGINGVERBINTENIS"
PARTNERSHIP_TYPE = "TYPEVERBINTENIS"

# Constants for specific values
DISSOLUTION_REASON_DIVORCE = "S"
PARTNERSHIP_TYPE_MARRIAGE = "H"
PARTNERSHIP_TYPE_REGISTERED = "P"
END_DATA_NA = "88888888"

# Hard-coded variables
START_YEAR = 2017  # Replace with your desired start year

PARTNERSHIP_FILE_PATH = "data/raw_data/GBAVERBINTENISPARTNER2023BUSV1.sav"
OUTPUT_FILE_PATH = "data/divorce_after_2016.csv"

def main():
    """Main function to execute the data preprocessing steps."""
    # Read data
    data = read_data()

    # Clean data
    data = clean_data(data)

    # Check partnerships ending in divorce
    check_divorces(data)

    # Process data according to START_YEAR
    final_data = process_data(data, START_YEAR)

    # Check for partnership duplicates
    check_partnership_duplicates(final_data)

    assert final_data[PERSON_ID].is_unique(), f"{PERSON_ID} is not unique in the final data"
    assert final_data[PARTNER_ID].is_unique(), f"{PARTNER_ID} is not unique in the final data"
    
    # Save final data to CSV
    final_data.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"Final data saved to {OUTPUT_FILE_PATH}")

def read_data():
    """Reads the input data from a .sav file and returns a DataFrame.

    Returns:
        A pandas DataFrame containing the loaded data.
    """
    # Read the data
    df, meta = pyreadstat.read_sav(PARTNERSHIP_FILE_PATH)

    # Convert columns to int
    df[PERSON_ID] = df[PERSON_ID].astype(int)
    df[PARTNER_ID] = df[PARTNER_ID].astype(int)

    return df

def clean_data(df):
    """Cleans the data according to specified rules.

    Args:
        df: DataFrame containing the data.

    Returns:
        Cleaned DataFrame.
    """
    initial_rows = len(df)
    print(f"Initial number of rows: {initial_rows}")

    # drop all rows where duplicate is not present with the partner and main id swapped
    df_swapped = df.copy()
    df_swapped[[PERSON_ID, PARTNER_ID]] = df_swapped[[PARTNER_ID, PERSON_ID]]
    df = df.merge(df_swapped, on=list(df.columns), how='inner')

    # Filter partnerships of type 'H' or 'P'
    df = df[
      df[PARTNERSHIP_TYPE].isin(
        [PARTNERSHIP_TYPE_MARRIAGE, PARTNERSHIP_TYPE_REGISTERED]
      )
    ]
    print(f"Rows after filtering by partnership type 'H' or 'P': {len(df)}")

    # Remove partnerships that have an end date but the reason is not 'S'
    condition = (
        (df[END_DATE] != END_DATA_NA) &
        (df[DISSOLUTION_REASON] != DISSOLUTION_REASON_DIVORCE)
    )
    df = df[~condition]
    print(f"Rows after removing non-divorce ended partnerships: {len(df)}")

    return df

def check_divorces(df):
    """Finds and prints the number of partnerships ending in divorce for each type.

    Args:
        df: DataFrame containing the data.
    """
    # Filter for partnerships that ended in divorce
    divorce_df = df[df[DISSOLUTION_REASON] == DISSOLUTION_REASON_DIVORCE]

    # Group by partnership type
    marriage_divorces = divorce_df[
        divorce_df[PARTNERSHIP_TYPE] == PARTNERSHIP_TYPE_MARRIAGE
    ]
    registered_divorces = divorce_df[
        divorce_df[PARTNERSHIP_TYPE] == PARTNERSHIP_TYPE_REGISTERED
    ]

    print(f"Number of marriages that ended in divorce: {len(marriage_divorces)}")
    print(f"Number of registered partnerships that ended in divorce: "
          f"{len(registered_divorces)}")

def process_data(df, start_year):
    """Processes the data according to START_YEAR and creates the label.

    Args:
        df: DataFrame containing the data.
        start_year: The START_YEAR variable.

    Returns:
        DataFrame with final columns and labels.
    """
    # Convert date columns to datetime
    df[START_DATE] = pd.to_datetime(
        df[START_DATE], format='%Y%m%d', errors='coerce')
    df[END_DATE] = pd.to_datetime(
        df[END_DATE].replace(END_DATA_NA, '99991231'),
        format='%Y%m%d', 
        errors='coerce'
    )

    # Filter partnerships valid till the last day of START_YEAR - 1
    cutoff_date = pd.Timestamp(year=start_year - 1, month=12, day=31)
    valid_df = df[
      (df[START_DATE] <= cutoff_date) &
      (df[END_DATE] > cutoff_date) 
    ]
    print(f"Rows after filtering partnerships valid till {cutoff_date.date()}: "
          f"{len(valid_df)}")

    # Create label column
    divorce_after_label = f"divorce_after_{start_year - 1}"

    # Label partnerships
    valid_df[divorce_after_label] = valid_df.apply(
        lambda row: 1 if (
            row[END_DATE] > cutoff_date and
            row[END_DATE] != pd.Timestamp('9999-12-31')
        ) else 0,
        axis=1
    )
    print(
      f"Number of partnerships labeled as 1 (divorced) after {start_year - 1}: "
      f"{valid_df[divorce_after_label].sum()}"
    )


    print(
      f"Number of partnerships labeled as 0 (current partners) after {start_year - 1}: "
      f"{valid_df[divorce_after_label].eq(0).sum()}"
    )

    # Select final columns
    final_df = valid_df[[PERSON_ID, PARTNER_ID, divorce_after_label]]

    return final_df

def check_partnership_duplicates(df):
    """Asserts that every partnership is listed twice with swapped IDs.

    Args:
        df: DataFrame containing the data.
    """
    # Create sorted tuples of IDs
    df['sorted_ids'] = df.apply(
        lambda row: tuple(sorted((row[PERSON_ID], row[PARTNER_ID]))), axis=1
    )

    partnership_counts = df['sorted_ids'].value_counts()

    missing_pairs = partnership_counts[partnership_counts != 2]

    if len(missing_pairs) == 0:
        print("Every partnership is listed twice with swapped IDs.")
    else:
        print(f"Number of partnerships missing the swapped counterpart: "
              f"{len(missing_pairs)}")
        print("Error: Not all partnerships have the swapped counterpart.")
        print("-"*20)
        print("The erroneous pairs are:")
        print(missing_pairs)
    # Drop the 'sorted_ids' column
    df.drop(columns='sorted_ids', inplace=True)

if __name__ == "__main__":
    main()
