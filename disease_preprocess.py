import pandas as pd


def count_cases(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Counts the number of cases for each disease in the dataset and formats the data for forecasting with `make_disease_forecast()`.
    Args:
        df (pd.DataFrame): DataFrame containing the data with 'Date' and 'Disease' columns.
    '''

    disease_counts = df.groupby(
    ['Date', 'Disease']).size().reset_index(name='Cases')

    return disease_counts
