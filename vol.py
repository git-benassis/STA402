import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import STL

# Download data

spy = yf.download("SPY", interval="15m", period="max")
save_path ="SPY_daily.csv"
# spy.to_csv(save_path)
# spy = pd.read_csv(save_path, index_col=0, parse_dates=True)
print(spy.head())

# Data treatment

# Printing data function
def print_data(x,y,mode,name,title,x_title,y_title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode=mode,
    name=name
    ))
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode="x unified"
    )
    fig.show()
    return(0)
    
# Volume data
spy_vol = spy["Volume"].squeeze()
print(spy_vol.head()) 

print_data(spy_vol.index,spy_vol.values,'lines+markers','Volume',"SPY Intraday Volume (30-min)","Datetime","Volume")

#ACF and PACF
# Compute ACF and PACF
lags = 60
acf_vals = acf(spy_vol, nlags=lags)
pacf_vals = pacf(spy_vol, nlags=lags)

print_data(list(range(lags + 1)),acf_vals,'lines+markers','ACF',"ACF of SPY Intraday Volume","Lag","ACF")
print_data(list(range(lags + 1)),pacf_vals,'lines+markers','PACF',"PACF of SPY Intraday Volume","Lag","PACF")

# moyenne mobile
vol_lisse = spy_vol.rolling(window=126).mean()
print_data(spy_vol.index,vol_lisse.values,'lines','Smoothed Volume',"Smoothed SPY Intraday Volume (30-day rolling mean)","Datetime","Smoothed Volume")

# différentiation

vol_diff = spy_vol.diff()
print_data(vol_diff.index,vol_diff.values,'lines','Differenced Volume',"Differenced SPY Intraday Volume","Datetime","Differenced Volume")
