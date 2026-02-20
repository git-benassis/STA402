import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import STL
from sklearn.model_selection import train_test_split #pour séparer données train/test
from sklearn.metrics import mean_squared_error

# Download and load data

#spy = yf.download("SPY", interval="1d", period="max")
save_path ="SPY_daily.csv"
#spy.to_csv(save_path)
spy = pd.read_csv(save_path, index_col=0, parse_dates=True)
# print(spy.head())

# Printing data function
def plot_data(x,y,mode,name,title,x_title,y_title):
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

# Data treatment
# Extract Volume Data

spy_vol = spy["Volume"].squeeze()
# print(spy_vol.head())
# plot_data(spy_vol.index,spy_vol.values,'lines+markers','Volume',"SPY dayly Volume","Datetime","Volume")

log_vol = np.log(spy_vol)
# print(log_vol.head())
plot_data(spy_vol.index, log_vol.values,'lines','Log Volume',"SPY dayly Log Volume","Datetime","Log Volume")

def evaluation_model(y_test,y_pred):
    rmse=mean_squared_error(y_test,y_pred)
    res=np.mean(y_test-y_pred)
    diff_acf=acf(y_test-y_pred)
    return rmse,res,diff_acf

#Split train/test Time Series
vol_train=spy_vol['2020-12-31':'2024-12-31'] # Keep only a small part of the data for training (old data is irrelevant)
vol_test=spy_vol['2025-01-01':] # Trying to predict the last two years (seems a bit too long)
log_vol_train=log_vol['2020-12-31':'2024-12-31']
log_vol_test=log_vol['2025-01-01':]

def plot_train_test(vol_train, vol_test, title='Volume SPY Train vs Test'):
    #fig = make_subplots(sizes=[1], subplot_titles=[title])
    fig=go.Figure()
    # Ajout trace train en bleu
    fig.add_trace(
        go.Scatter(
            x=vol_train.index, 
            y=vol_train.values, 
            mode='lines',
            name='Train (jusqu\'au 2023-12-31)',
            line=dict(color='blue', width=2)
        )
    )
    
    # Ajout trace test en orange
    fig.add_trace(
        go.Scatter(
            x=vol_test.index, 
            y=vol_test.values, 
            mode='lines',
            name='Test (2024-01-01 et après)',
            line=dict(color='orange', width=2)
        )
    )
    
    # Layout
    fig.update_layout(
        title={'text': title, 'font': {'size': 16}},
        xaxis_title='Date',
        yaxis_title='Volume',
        hovermode='x unified',
        template='plotly_white'
    )
    
    # Grille et rotation (optionnel)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', tickangle=45)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    fig.show()

# Graphic representation of train/test split
# plot_train_test(vol_train, vol_test)
# plot_train_test(log_vol_train, log_vol_test, title='Log Volume SPY Train vs Test')

# Value distribution

def plot_distribution(series, title='Distribution', x_title='Value', y_title='Frequency'):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=series.values, nbinsx=50, name='Distribution', marker_color='blue'))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, template='plotly_white')
    fig.show()

plot_distribution(spy_vol, title='Distribution of SPY Daily Volume', x_title='Volume', y_title='Frequency')
plot_distribution(log_vol, title='Distribution of SPY Daily Log Volume', x_title='Log Volume', y_title='Frequency')

# ACF and PACF
# Compute ACF and PACF
lags = 63 # equivalent to 3 months of trading days (assuming 21 trading days per month )
acf_vals = acf(vol_train, nlags=lags)
pacf_vals = pacf(vol_train, nlags=lags)
acf_vals_log = acf(log_vol_train, nlags=lags)
pacf_vals_log = pacf(log_vol_train, nlags=lags)

# plot_data(list(range(lags + 1)),acf_vals,'lines+markers','ACF',"ACF of SPY dayly Volume","Lag","ACF")
# plot_data(list(range(lags + 1)),pacf_vals,'lines+markers','PACF',"PACF of SPY dayly Volume","Lag","PACF")
# plot_data(list(range(lags + 1)),acf_vals_log,'lines+markers','ACF',"ACF of SPY dayly Log Volume","Lag","ACF")
# plot_data(list(range(lags + 1)),pacf_vals_log,'lines+markers','PACF',"PACF of SPY dayly Log Volume","Lag","PACF")

# moyenne mobile
vol_lisse = vol_train.rolling(window=21).mean() # rolling mean over 1 month
# plot_data(vol_lisse.index,vol_lisse.values,'lines','Smoothed Volume',"Smoothed SPY dayly Volume (monthly rolling mean)","Datetime","Smoothed Volume")

# différentiation
vol_diff = vol_train.diff() # first order differentiation
# plot_data(vol_diff.index,vol_diff.values,'lines','Differenced Volume',"Differenced SPY dayly Volume","Datetime","Differenced Volume")

# Seasonal Observation
def seasonal_cobweb(series, freq="month", title="Seasonality Cobweb"):
    """
    series : pd.Series with DatetimeIndex
    freq   : "month", "dayofweek", "week", "hour"
    """

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series must have a DatetimeIndex")

    # ---- Aggregation depending on frequency ----
    if freq == "month":
        grouped = series.groupby(series.index.month).mean()
        labels = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]

    elif freq == "dayofweek":
        grouped = series.groupby(series.index.dayofweek).mean()
        labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

    elif freq == "week":
        grouped = series.groupby(series.index.isocalendar().week).mean()
        labels = grouped.index.astype(str)

    elif freq == "hour":
        grouped = series.groupby(series.index.hour).mean()
        labels = grouped.index.astype(str)

    else:
        raise ValueError("freq must be: month, dayofweek, week, hour")

    values = grouped.values

    # Close the loop
    values = np.append(values, values[0])
    labels = list(labels) + [labels[0]]

    # ---- Plotly Radar ----
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name="Mean"
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title=title,
        showlegend=False
    )

    fig.show()

# seasonal_cobweb(spy_vol, freq="month", title="Seasonality Cobweb - Monthly")
# seasonal_cobweb(spy_vol, freq="dayofweek", title="Seasonality Cobweb - Day of Week")
# seasonal_cobweb(spy_vol, freq="week", title="Seasonality Cobweb - Week of Year")

# Trend observation and estimation

vol_trend = spy_vol.rolling(window=252).mean() # rolling mean over 1 year
# plot_data(vol_trend.index,vol_trend.values,'lines','Trend Volume',"SPY Volume - Trend (252-day rolling mean)","Datetime","Volume")

# Lissage exponentiel simple 
alpha=0.9
def holt_forecast(vol_train, n_forecast, alpha, beta):
    """
    Holt Double Exponential Smoothing (niveau + tendance) - CORRIGÉ
    """
    # Initialisation (sur train)
    level = vol_train.iloc[-1]
    trend = vol_train.diff().tail(10).mean() or 0  # Moyenne récente diffs
    
    forecast = []
    current_level = level
    current_trend = trend
    
    for i in range(n_forecast):
        # 1. PRÉDICTION h=1
        pred_h1 = current_level + (i + 1) * current_trend  # Linéaire
        
        forecast.append(pred_h1)
        
        # 2. MISE À JOUR (seulement si on avait des données réelles)
        # Pour pure forecast, on skippe l'update ou on approxime
        # Ici on garde constantes les params (simple forecast)
    
    return pd.Series(forecast, index=vol_test.index)

# Usage
alpha, beta = 0.5, 0.1
holt_pred = holt_forecast(vol_train, len(vol_test), alpha, beta)


# Usage

def plot_predictions(vol_test, predictions_list, pred_names, title='Prédictions Volume SPY Test'):
    """
    vol_test: données test réelles (pd.Series)
    predictions_list: liste de pd.Series ou arrays de prédictions (même longueur que vol_test)
    pred_names: liste de noms pour légende
    """
    fig = go.Figure()
    
    # Données réelles (noir)
    fig.add_trace(go.Scatter(
        x=vol_test.index,
        y=vol_test.values,
        mode='lines',
        name='Données Test Réelles',
        line=dict(color='black', width=3)
    ))
    
    # Toutes les prédictions
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    for i, (pred, name) in enumerate(zip(predictions_list, pred_names)):
        fig.add_trace(go.Scatter(
            x=vol_test.index,
            y=pred.values if hasattr(pred, 'values') else pred,
            mode='lines',
            name=name,
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Volume',
        hovermode="x unified",
        template='plotly_white'
    )
    fig.show()

prediction_list=[holt_pred]
prediction_names=[f"lissage de Holt Winters, alpha={alpha} et beta={beta}"]
plot_predictions(vol_test,prediction_list,prediction_names)
erreur_quadratique, res, diff_acf = evaluation_model(vol_test,holt_pred)
print(erreur_quadratique,res)
# plot_data(diff_acf)
# il faut maintenant comparer avec la courbe réelle pour voir si les prédictions sont bonnes 