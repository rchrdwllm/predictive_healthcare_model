import pandas as pd
import numpy as np
from disease_preprocess import count_cases
from disease_forecast import make_disease_forecast


def detect_outbreak(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Detects disease outbreaks in a given DataFrame
    Args:
        df (pd.DataFrame): DataFrame containing disease data with 'Date', 'Disease', and 'Cases' columns.
    Returns:
        pd.DataFrame: DataFrame with an additional column 'Outbreak' indicating outbreak status.
    '''
    df = df.copy()
    # Calculate average cases per prognosis
    avg_cases = df.groupby('Disease')['Cases'].transform('mean')
    # Outbreak if cases >= 2x average for that prognosis
    df['Outbreak'] = (df['Cases'] >= 2 * avg_cases).astype(int)

    return df


def detect_outbreak_per_day(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    '''
    Detects outbreaks per prognosis per day using the rolling mean of previous days (excluding the current day).
    Args:
        df (pd.DataFrame): DataFrame with 'Date', 'Disease', and 'Cases' columns.
        window (int): Number of previous days to use for the rolling mean.
    Returns:
        pd.DataFrame: DataFrame with 'Date', 'Disease', 'Cases', 'Outbreak'.
    '''
    df = df.copy()

    daily = df.groupby(['Date', 'Disease'], as_index=False)['Cases'].sum()
    daily = daily.sort_values(['Disease', 'Date'])

    daily['rolling_mean_prev'] = daily.groupby('Disease')['Cases'].apply(
        lambda x: x.rolling(window, min_periods=1).mean().shift(1)
    ).reset_index(level=0, drop=True)

    # Only flag outbreak if rolling_mean_prev > 0 and Cases > 0
    daily['Outbreak'] = ((daily['rolling_mean_prev'].notna()) &
                         (daily['rolling_mean_prev'] > 0) &
                         (daily['Cases'] > 0) &
                         (daily['Cases'] >= 2 * daily['rolling_mean_prev'])).astype(int)
    daily = daily.drop(columns=['rolling_mean_prev'])

    return daily


def predict_future_outbreaks(df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    '''
    Predicts future outbreaks based on the last known Cases.
    Args:
        df (pd.DataFrame): DataFrame with 'Date', 'Disease', and 'Cases' columns.
        days (int): Number of days to predict.
    Returns:
        pd.DataFrame: DataFrame with predicted outbreaks.
    '''

    df = df.copy()
    historical_means = df.groupby('Disease')['Cases'].mean()

    forecasts = make_disease_forecast(df, period=days)

    results = []

    for forecast in forecasts:
        prognosis = forecast['Disease'].unique()[0]

        forecast = forecast.rename(
            columns={'ds': 'Date', 'yhat': 'Cases'})

        outbreak_predictions = detect_outbreak_per_day(forecast)

        results.append(outbreak_predictions)

    return pd.concat(results, ignore_index=True)
