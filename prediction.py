import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA, sarimax
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import arma_order_select_ic

def arima_forecast(train:pd.Series, test:pd.Series, order=(1,1,1)):
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    return forecast

def sarimax_forecast(train:pd.Series, test:pd.Series, order=(1,1,1), seasonal_order=(0,0,0,0)):
    model = sarimax.SARIMAX(train, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    return forecast

def prophet_forecast(train:pd.Series, test:pd.Series):
    # Prepare the series
    df = pd.DataFrame({
        'ds': train.index,
        'y': train.values
    })

    model = Prophet(weekly_seasonality=True, daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=len(test), freq='B')
    forecast = model.predict(future)
    y_pred = forecast['yhat'].iloc[-len(test):].values
    return y_pred

def evaluate_forecast(test:pd.Series, forecast:pd.Series):
    rmse = np.sqrt(mean_squared_error(test, forecast))
    return rmse

def exogenous_forecast(train:pd.Series, test:pd.Series, exog_train:pd.DataFrame, exog_test:pd.DataFrame, order=(1,1,1), seasonal_order=(0,0,0,5)):
    model = sarimax.SARIMAX(train, exog=exog_train, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test), exog=exog_test)
    return forecast

def holt_forecast(train: pd.Series, test: pd.Series, trend='add', seasonal=None, seasonal_periods=5):    
    model = ExponentialSmoothing(
        train,
        trend=trend,                
        seasonal=seasonal,             
        seasonal_periods=seasonal_periods
    )
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    return forecast

