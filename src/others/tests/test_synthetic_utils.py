
import pandas as pd 
import numpy as np 
import pyreadstat 
import pytest

import others.synthetic_data_generation.utils as su

@pytest.fixture
def wage_df():
    "dataframe with 2 observations per person"
    df = pd.DataFrame({
        "person": [1.,2.,3.,4.,1.,2.,3.,4.], 
        "spell": [1.,1.,1.,1.,2.,2.,2.,2.], 
        "wage": np.random.rand(8)}
    )
    return df


@pytest.fixture
def wage_sav_file(wage_df, tmp_path):
    filepath = tmp_path / "wage_df.sav"
    pyreadstat.write_sav(wage_df, filepath)
    return filepath

@pytest.fixture
def wage_csv_file(wage_df, tmp_path):
    filepath = tmp_path / "wage_df.csv"
    wage_df.to_csv(filepath, index=False)
    return filepath

@pytest.fixture
def wage_csv_file_with_semicolon(wage_df, tmp_path):
    filepath = tmp_path / "wage_df_semicolon.csv"
    wage_df.to_csv(filepath, index=False, sep=";")
    return filepath


def test_check_column_names():
    # Test case 1: Basic functionality
    column_names = ['apple', 'banana', 'grape', 'pineapple', 'watermelon']
    names_to_check = ['apple', 'banana']
    assert su.check_column_names(column_names, names_to_check)

    # Test case 2: No matches
    column_names = ['kiwi', 'orange', 'pear']
    names_to_check = ['apple', 'banana']
    assert not su.check_column_names(column_names, names_to_check)

    # Test case 3: Partial matches
    column_names = ['apple', 'banana', 'grape', 'pineapple', 'watermelon']
    names_to_check = ['ppl', 'anana']
    assert su.check_column_names(column_names, names_to_check) 

    # Test case 4: Empty input lists
    column_names = []
    names_to_check = []
    assert not su.check_column_names(column_names, names_to_check)


def test_subsample_from_ids(wage_df):
    sampled_df = su.subsample_from_ids(wage_df, id_col="person", frac=0.5)
    assert sampled_df.shape == (4, wage_df.shape[1]), "does not return right shape"
    assert (sampled_df.columns == wage_df.columns).all(), "does not retain column names"
    assert sampled_df["person"].nunique() == 2, "does not sample right fraction of ids"


def test_sample_from_file(wage_df, wage_sav_file, wage_csv_file, wage_csv_file_with_semicolon):
    nrow = 3
    df, n = su.sample_from_file(str(wage_sav_file), nrow)
    assert n == wage_df.shape[0], "sav returns wrong size of table"
    pd.testing.assert_frame_equal(df, wage_df.loc[:nrow-1, :]), "sav returns wrong subsampled df"

    df, n = su.sample_from_file(str(wage_csv_file), nrow)
    assert n == wage_df.shape[0], "csv returns wrong size of table"
    pd.testing.assert_frame_equal(df, wage_df.loc[:nrow-1, :]), "csv returns wrong subsampled df"

    df, n = su.sample_from_file(str(wage_csv_file_with_semicolon), nrow)
    assert n == wage_df.shape[0], "csv returns wrong size of table"
    pd.testing.assert_frame_equal(df, wage_df.loc[:nrow-1, :]), "csv returns wrong subsampled df"

    with pytest.raises(ValueError, match="wrong path"): 
        su.sample_from_file("wrong path", nrow)

