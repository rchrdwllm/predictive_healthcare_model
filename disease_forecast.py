from prophet import Prophet
import pandas as pd
from disease_preprocess import count_cases


def make_disease_forecast(df: pd.DataFrame, period: int = 7) -> list:
    '''
    Forecasts the number of cases for each disease in the dataset for a given period.
    Args:
        df (pd.DataFrame): DataFrame containing the data with 'Date' and 'Disease' columns.
        period (int): Number of days to forecast into the future.
    Returns:
        forecasts (list): List of DataFrames containing the forecasted cases for each disease.
    '''

    unique_diseases = df['Disease'].unique()

    forecasts = []

    for disease in unique_diseases:
        print(f"Forecasting for {disease}...")

        disease_data = df[df['Disease'] == disease]
        disease_data = disease_data.rename(
            columns={'Date': 'ds', 'Cases': 'y'})

        if disease_data.dropna().shape[0] < 2:
            print(f"Not enough data to forecast for {disease}.")

            continue

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.1,
        )
        model.fit(disease_data)

        future = model.make_future_dataframe(periods=period)
        forecast = model.predict(future)
        forecast['Disease'] = disease
        forecast['yhat'] = forecast['yhat'].round(0).astype(int)

        forecast = forecast[['ds', 'yhat', 'Disease']]
        forecast = forecast.rename(
            columns={'ds': 'Date', 'yhat': 'Cases'})

        forecast['Cases'] = forecast['Cases'].clip(lower=0)
        forecast['Date'] = pd.to_datetime(forecast['Date'])
        last_hist_date = disease_data['ds'].max()
        historical = disease_data.rename(columns={'ds': 'Date', 'y': 'Cases'})[
            ['Date', 'Cases']]
        historical['Disease'] = disease
        historical['Date'] = pd.to_datetime(historical['Date'])
        forecast = forecast[forecast['Date'] >
                            last_hist_date].reset_index(drop=True)
        combined = pd.concat([historical, forecast], ignore_index=True)
        forecasts.append(combined)

    return forecasts
