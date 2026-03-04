import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf, pacf

def log_volume(ts:pd.Series):
    return np.log(ts)

def acf_pacf(ts:pd.Series, lags:int=63):
    return acf(ts, nlags=lags), pacf(ts, nlags=lags)

def key_indicators(ts:pd.Series):
    return {
        "mean": ts.mean(),
        "std": ts.std(),
        "min": ts.min(),
        "25%": ts.quantile(0.25),
        "50%": ts.median(),
        "75%": ts.quantile(0.75),
        "max": ts.max()
    }

def moving_average(ts:pd.Series, window:int=21):
    return ts.rolling(window=window).mean()

def difference(ts:pd.Series, order:int=1):
    return ts.diff(periods=order).dropna()

def trend(ts:pd.Series):
    return moving_average(ts, 252)

def anomalies(ts:pd.Series, window:int=252, threshold:float=2.0):
    ma = moving_average(ts, window)
    residuals = ts - ma
    anomalies = residuals[abs(residuals) > threshold * ma]
    return anomalies

def partition(ts:pd.Series, freq:str="month"):
    if freq == "month":
        return ts.groupby(ts.index.month).mean()
    elif freq == "dayofweek":
        return ts.groupby(ts.index.dayofweek).mean()
    elif freq == "week":
        return ts.groupby(ts.index.isocalendar().week).mean()
    elif freq == "hour":
        return ts.groupby(ts.index.hour).mean()
    elif freq == "day":
        return ts.groupby(ts.index.day).mean()
    elif freq == "dayofyear":
        return ts.groupby(ts.index.dayofyear).mean()
    else:
        raise ValueError("freq must be: month, dayofweek, week, hour")


def compute_exogenous_variables(df:pd.DataFrame):
    df = df.copy()

    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df["abs_ret"] = df["log_ret"].abs()
    df["sq_ret"] = df["log_ret"] ** 2

    df["roll_vol_5"] = df["log_ret"].rolling(5).std()
    df["roll_vol_10"] = df["log_ret"].rolling(10).std()

    df["range"] = np.log(df["High"]) - np.log(df["Low"])

    df["gap"] = np.log(df["Open"] / df["Close"].shift(1))

    exog_cols = [
        "abs_ret",
        "sq_ret",
        "roll_vol_5",
        "roll_vol_10",
        "range",
        "gap",
    ]

    exog = df[exog_cols]
    exog = exog.shift(1)
    exog = exog.dropna()

    return exog