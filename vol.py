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

spy_vol = spy["Volume"].squeeze()
print(spy_vol.head()) 

# Print data

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=spy_vol.index,
    y=spy_vol.values,
    mode='lines+markers',
    name='Volume'
))
fig.update_layout(
    title="SPY Intraday Volume (30-min)",
    xaxis_title="Datetime",
    yaxis_title="Volume",
    hovermode="x unified"
)
fig.show()

# ACF and PACF

# Compute ACF and PACF
lags = 60
acf_vals = acf(spy_vol, nlags=lags)
pacf_vals = pacf(spy_vol, nlags=lags)

# Plot ACF and PACF

fig_acf = go.Figure()
fig_acf.add_trace(go.Scatter(
    x=list(range(lags + 1)),
    y=acf_vals,
    mode='lines+markers',
    name='ACF'
))
fig_acf.update_layout(
    title="ACF of SPY Intraday Volume",
    xaxis_title="Lag",
    yaxis_title="ACF",
    hovermode="x unified"
)
fig_acf.show()

fig_pacf = go.Figure()
fig_pacf.add_trace(go.Scatter(
    x=list(range(lags + 1)),
    y=pacf_vals,
    mode='lines+markers',
    name='PACF'
))
fig_pacf.update_layout(
    title="PACF of SPY Intraday Volume",
    xaxis_title="Lag",
    yaxis_title="PACF",
    hovermode="x unified"
)
fig_pacf.show()

vol_lisse = spy_vol.rolling(window=126).mean()

fig_lisse = go.Figure()
fig_lisse.add_trace(go.Scatter(
    x=spy_vol.index,
    y=vol_lisse.values,
    mode='lines',
    name='Smoothed Volume'
))
fig_lisse.update_layout(
    title="Smoothed SPY Intraday Volume (30-day rolling mean)",
    xaxis_title="Datetime",
    yaxis_title="Smoothed Volume",
    hovermode="x unified"
)
fig_lisse.show()

vol_diff = spy_vol.diff()
fig_diff = go.Figure()
fig_diff.add_trace(go.Scatter(
    x=vol_diff.index,
    y=vol_diff.values,
    mode='lines',
    name='Differenced Volume'
))
fig_diff.update_layout(
    title="Differenced SPY Intraday Volume",
    xaxis_title="Datetime",
    yaxis_title="Differenced Volume",
    hovermode="x unified"
)
fig_diff.show()