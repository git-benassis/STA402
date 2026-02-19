import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import STL

# Download data

#spy = yf.download("SPY", interval="15m", period="max")
save_path ="SPY_daily.csv"
#spy.to_csv(save_path)
spy = pd.read_csv(save_path, index_col=0, parse_dates=True)
print(spy.head())

# Data treatment

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
    
# Volume data
spy_vol = spy["Volume"].squeeze()
print(spy_vol.head()) 

plot_data(spy_vol.index,spy_vol.values,'lines+markers','Volume',"SPY Intraday Volume (30-min)","Datetime","Volume")

"""#ACF and PACF
# Compute ACF and PACF
lags = 60
acf_vals = acf(spy_vol, nlags=lags)
pacf_vals = pacf(spy_vol, nlags=lags)

plot_data(list(range(lags + 1)),acf_vals,'lines+markers','ACF',"ACF of SPY Intraday Volume","Lag","ACF")
plot_data(list(range(lags + 1)),pacf_vals,'lines+markers','PACF',"PACF of SPY Intraday Volume","Lag","PACF")

# moyenne mobile
vol_lisse = spy_vol.rolling(window=126).mean()
plot_data(spy_vol.index,vol_lisse.values,'lines','Smoothed Volume',"Smoothed SPY Intraday Volume (30-day rolling mean)","Datetime","Smoothed Volume")

# différentiation
vol_diff = spy_vol.diff()
plot_data(vol_diff.index,vol_diff.values,'lines','Differenced Volume',"Differenced SPY Intraday Volume","Datetime","Differenced Volume")
""""""
# estimation de la tendance

x_num = np.arange(len(spy_vol.dropna()))  # Index numérique pour régression
y_clean = spy_vol.dropna().values

coeff = np.polyfit(x_num, y_clean, 100)  # régression par polynome de degré 100
p_trend = np.poly1d(coeff)
y_trend = p_trend(x_num)

def plot_data_trend(x, y, x_trend, y_trend, mode, name_data, name_trend, title, x_title, y_title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode=mode, name=name_data))
    fig.add_trace(go.Scatter(x=x_trend, y=y_trend, mode='lines', name=name_trend, line=dict(color='red', width=3)))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, hovermode="x unified")
    fig.show()
    return 0

# Appel
plot_data_trend(spy_vol.index[1:], y_clean, spy_vol.index[1:], y_trend, 'lines+markers', 'Differenced Volume', 'Tendance linéaire', "SPY Volume Diff + Tendance", "Datetime", "Volume Diff")
"""
"""# Moyenne mobile en prenant en compte la saisonnalité
# Période saisonnière (ex: 26 périodes 15min = 6h30, 1/2 session trading)
period_saison = 26  

# Moyenne mobile saisonnière (sur 1 cycle complet)
vol_saison = spy_vol.rolling(window=period_saison, center=True).mean()"""

# Composante saisonnière = moyenne mobile sur la période

# Données désaisonnalisées
vol_desaisson = spy_vol - vol_saison

# Visualisation avec votre fonction
plot_data(spy_vol.index, vol_saison.values, 'lines', 'Saisonnalité', 
           "SPY Volume - Composante Saisonnière (MM 26p)", "Datetime", "Volume")
           
plot_data(spy_vol.index, vol_desaisson.values, 'lines', 'Désaisonnalisé', 
           "SPY Volume - Données Désaisonnalisées", "Datetime", "Volume Désaison.")


# Lissage exponentiel simple 
alpha = 0.5  
vol_ewma = spy_vol.ewm(alpha=alpha, adjust=True).mean()

plot_data(spy_vol.index, vol_ewma.values, 'lines', 'EWMA', 
           f"SPY Volume - Lissage Exponentiel (α={alpha})", "Datetime", "Volume Lissé")

# il faut maintenant comparer avec la courbe réelle pour voir si les prédictions sont bonnes 
