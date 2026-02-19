import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import STL
from sklearn.model_selection import train_test_split #pour séparer données train/test
from sklearn.metrics import mean_squared_error

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

def evaluation_model(y_test,y_pred):
    rmse=mean_squared_error(y_test,y_pred)
    res=np.mean(y_test-y_pred)
    diff_acf=acf(y_test-y_pred)
    return rmse,res,diff_acf
    
# Volume data
spy_vol = spy["Volume"].squeeze()
print(spy_vol.head()) 

vol_train=spy_vol['2020-12-31':'2023-12-31']
vol_test=spy_vol['2024-01-01':]


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

# Utilisation directe
plot_train_test(vol_train, vol_test)


#plot_data(spy_vol.index,spy_vol.values,'lines+markers','Volume',"SPY Intraday Volume (30-min)","Datetime","Volume")

#ACF and PACF
# Compute ACF and PACF
# lags = 60
# acf_vals = acf(vol_train, nlags=lags)
# pacf_vals = pacf(vol_train, nlags=lags)

# plot_data(list(range(lags + 1)),acf_vals,'lines+markers','ACF',"ACF of SPY Intraday Volume","Lag","ACF")
# plot_data(list(range(lags + 1)),pacf_vals,'lines+markers','PACF',"PACF of SPY Intraday Volume","Lag","PACF")

# # moyenne mobile
# vol_lisse = vol_train.rolling(window=22).mean()
# plot_data(vol_lisse.index,vol_lisse.values,'lines','Smoothed Volume',"Smoothed SPY Intraday Volume (30-day rolling mean)","Datetime","Smoothed Volume")

# # différentiation
# vol_diff = vol_train.diff()
# plot_data(vol_diff.index,vol_diff.values,'lines','Differenced Volume',"Differenced SPY Intraday Volume","Datetime","Differenced Volume")


# # estimation de la tendance
# x_num = np.arange(len(spy_vol.dropna()))  # Index numérique pour régression
# y_clean = spy_vol.dropna().values

# coeff = np.polyfit(x_num, y_clean, 100)  # régression par polynome de degré 100
# p_trend = np.poly1d(coeff)
# y_trend = p_trend(x_num)

# def plot_data_trend(x, y, x_trend, y_trend, mode, name_data, name_trend, title, x_title, y_title):
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=x, y=y, mode=mode, name=name_data))
#     fig.add_trace(go.Scatter(x=x_trend, y=y_trend, mode='lines', name=name_trend, line=dict(color='red', width=3)))
#     fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, hovermode="x unified")
#     fig.show()
#     return 0

# # Appel
# plot_data_trend(spy_vol.index[1:], y_clean, spy_vol.index[1:], y_trend, 'lines+markers', 'Differenced Volume', 'Tendance linéaire', "SPY Volume Diff + Tendance", "Datetime", "Volume Diff")

# # Moyenne mobile en prenant en compte la saisonnalité
# # Période saisonnière (ex: 26 périodes 15min = 6h30, 1/2 session trading)
# period_saison = 26  

# # Moyenne mobile saisonnière (sur 1 cycle complet)
# vol_saison = spy_vol.rolling(window=period_saison, center=True).mean()

# # Composante saisonnière = moyenne mobile sur la période

# # Données désaisonnalisées
# vol_desaisson = spy_vol - vol_saison

# # Visualisation avec votre fonction
# plot_data(spy_vol.index, vol_saison.values, 'lines', 'Saisonnalité', 
#            "SPY Volume - Composante Saisonnière (MM 26p)", "Datetime", "Volume")
           
# plot_data(spy_vol.index, vol_desaisson.values, 'lines', 'Désaisonnalisé', 
#            "SPY Volume - Données Désaisonnalisées", "Datetime", "Volume Désaison.")


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
plot_data(diff_acf)
# il faut maintenant comparer avec la courbe réelle pour voir si les prédictions sont bonnes 