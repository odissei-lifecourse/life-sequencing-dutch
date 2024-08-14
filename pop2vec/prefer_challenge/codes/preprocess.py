import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
import time 
from tqdm import tqdm
import datatable as dt

TARGET = 'new_child'
KEY = 'nomem_encr'
def preprocess_background_data(background_data_path, dtype_mapping):
    # Load the background data
    background_data = pd.read_csv(background_data_path, dtype=dtype_mapping)

    # Convert 'wave' to str if not already, ensuring sorting works as expected
    background_data['wave'] = background_data['wave'].astype(str)

    # Sort by KEY and 'wave' in descending order
    background_data.sort_values(
        by=[KEY, 'wave'], 
        ascending=[True, False], 
        inplace=True
    )

    # Aggregate using the first non-null value in each group
    def first_non_null(series):
        return series.dropna().iloc[0] if not series.dropna().empty else pd.NA

    # Group by KEY
    background_data_latest = background_data.groupby(
        KEY
    ).agg(first_non_null)

    # Reset index to undo the grouping effect
    background_data_latest.reset_index(inplace=True)

    # Drop the 'wave' column
    background_data_latest.drop('wave', axis=1, inplace=True)

    return background_data_latest

# Now, you can proceed to merge this processed background data with your train data

def get_dtype_mapping(codebook):
    dtype_mapping = {}
    for _, row in tqdm(codebook.iterrows()):
        if row['type_var'] == 'numeric':
            dtype_mapping[row['var_name']] = 'float32'
        else:# row['type_var'] == 'categorical':
            dtype_mapping[row['var_name']] = 'str'
    return dtype_mapping

def load_data(nrows=None, col_subset=None):
    
    codebook = pd.read_csv('data/PreFer_codebook.csv')






    dtype_mapping = get_dtype_mapping(codebook)
    train_background = preprocess_background_data(
        'data/PreFer_train_background_data.csv',
        dtype_mapping
    )
    if nrows is not None:
        train_background = train_background.iloc[:nrows]
    
    train_outcome = pd.read_csv(
        'data/PreFer_train_outcome.csv', 
        nrows=nrows,
        dtype=dtype_mapping,
    )
    train_data = pd.read_csv(
        'data/PreFer_train_data.csv',
        nrows=nrows,
        dtype=dtype_mapping
    )
    if col_subset is not None:
        codebook = codebook.sort_values(by=['prop_missing'])
        top_cols = codebook[
            codebook['dataset']=='PreFer_train_data.csv'
        ]['var_name'].iloc[:col_subset]
        train_data = train_data[top_cols]
    return train_outcome, train_data, train_background, codebook

def merge_data(train_data, train_background, train_outcome):
    train_combined = train_data.merge(
        train_background, on=KEY, how='left'
    )
    train_combined = train_combined.merge(
        train_outcome, on=KEY, how='left'
    )
    return train_combined

def encode_and_clean_data(train_combined, codebook):
    # Classify columns based on their type_var from the codebook
    categorical_vars = (
        codebook[codebook['type_var'] == 'categorical']['var_name']
    )
    categorical_vars = [col for col in categorical_vars if col in train_combined.columns]

    open_ended_vars = (
        codebook[codebook['type_var'] == 'response to open-ended question']
        ['var_name']
    )
    open_ended_vars = [col for col in open_ended_vars if col in train_combined.columns]
    
    print("NOW1")

    character_condition = (
        codebook['type_var'] == 'character [almost exclusively empty strings]'
    )
    date_time_condition = codebook['type_var'] == 'date or time'
    ignore_vars = (
        codebook[character_condition | date_time_condition]['var_name']
    )
    ignore_vars = [col for col in ignore_vars if col in train_combined.columns]
    
    print("NOW2")

    # Drop columns that need to be ignored
    train_combined.drop(ignore_vars, axis=1, inplace=True)

    # st = time.time()
    # # Encode open-ended responses as binary
    for col in tqdm(open_ended_vars):
        train_combined[col] = train_combined[col].notna().astype(int)
    

    # for col in tqdm(categorical_vars):
    #     train_combined[col] = train_combined[col].astype(str)

    # print(f"{time.time()-st} seconds for dtype fixing")
    
    # Which categorical variables to one-hot encode and which to target encode
    max_categories_for_one_hot = 15
    low_cardinality_cats = (
        codebook[
            (codebook['unique_values_n'] <= max_categories_for_one_hot) & 
            (codebook['type_var'] == 'categorical')
        ]['var_name']
    )
    low_cardinality_cats = [col for col in low_cardinality_cats if col in train_combined.columns]

    high_cardinality_cats = (
        codebook[
            (codebook['unique_values_n'] > max_categories_for_one_hot) & 
            (codebook['type_var'] == 'categorical')
        ]['var_name']
    )
    high_cardinality_cats = [col for col in high_cardinality_cats if col in train_combined.columns]

    # Initialize encoders
    oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    target_encoder = TargetEncoder()
    print("YOLO")
    # Apply one-hot encoding
    if len(low_cardinality_cats) > 0:
        st = time.time()
        one_hot_encoded = pd.DataFrame(
            oh_encoder.fit_transform(
                train_combined[low_cardinality_cats].fillna('Missing')
            ), 
            columns=oh_encoder.get_feature_names_out(low_cardinality_cats)
        )
        print(f"{time.time()-st} seconds for ohe")
        train_combined.drop(low_cardinality_cats, axis=1, inplace=True)
        train_combined = pd.concat([train_combined, one_hot_encoded], axis=1)
        print(f"{time.time()-st} seconds for ohe total")
    print("1YOLO")
    # Apply target encoding
    if len(high_cardinality_cats) > 0:
        st = time.time()
        train_combined[high_cardinality_cats] = target_encoder.fit_transform(
            train_combined[high_cardinality_cats].fillna('Missing'), 
            train_combined[TARGET]
        )  
        print(f"{time.time()-st} seconds for te")
        
    return train_combined

def fill_missing_with_mean(df):
    # Compute the mean for each numeric column
    means = df.drop(columns=[TARGET, KEY]).mean()
    
    # Fill missing values with the computed means for all columns except the 
    # excluded ones
    df.update(df.drop(columns=[TARGET, KEY]).fillna(means))
    return df

def save_using_dt(df, path):
    st = time.time()
    frame = dt.Frame(df)
    frame.to_csv(path, verbose=True)
    end = time.time()
    print(f"dt save time: {end-st} seconds")

def keep_rows_with_outcome(df):
    df_with_outcome = df


def main():
    train_outcome, train_data, train_background, codebook = load_data(200, 5000)
    print("H1")
    train_combined = merge_data(train_data, train_background, train_outcome)
    print("H2")
    train_combined = encode_and_clean_data(train_combined, codebook)
    print("H3")
    print(train_combined.dtypes)
    
    # Convert KEY to string
    train_combined[KEY] = train_combined[KEY].astype(str)

    # # Convert all other columns to float32, except KEY
    # for col in tqdm(train_combined.columns.drop(KEY)):
    #     train_combined[col] = train_combined[col].astype('float32')

    # save_using_dt(
    #     train_combined, 
    #     'data/train_data_combined_before_imputing.csv',
    # )


    print(len(train_combined), "before drop")
    train_combined = train_combined.dropna(subset=[TARGET])
    print(len(train_combined), "after drop")
    cols_to_drop = train_combined.columns[train_combined.nunique() <= 1]
    print(f"cols to drop are: {cols_to_drop}")
    # Drop these columns
    train_combined = train_combined.drop(columns=cols_to_drop)
    # train_combined = train_combined.dropna(axis=1, how='all')
    # print("OKAYAAAAA!!!")
    train_combined = fill_missing_with_mean(train_combined)
    save_using_dt(
        train_combined, 
        'data/train_data_combined_after_imputing.csv',
    )

    # with open('data/final_train_data.csv', 'w', buffering=4194304) as f:  # buffer size of 4MB
    #     train_combined.to_csv(f, index=False)
    #train_combined.to_csv('data/final_train_data.csv', index=False)

    # Optionally, display the shape and types for verification
    print(train_combined.shape)
    print(train_combined.dtypes)

if __name__ == '__main__':
    main()
