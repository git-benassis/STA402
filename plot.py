import plotly.graph_objects as go
import pandas as pd
import numpy as np

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


def plot_distribution(series, title='Distribution', x_title='Value', y_title='Frequency'):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=series.values, nbinsx=50, name='Distribution', marker_color='blue'))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, template='plotly_white')
    fig.show()

def plot_acf_pacf(acf, pacf):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(acf))), y=acf, mode='markers', name='ACF'))
    fig.add_trace(go.Scatter(x=list(range(len(pacf))), y=pacf, mode='markers', name='PACF'))
    fig.update_layout(title='ACF and PACF', xaxis_title='Lag', yaxis_title='Value', template='plotly_white')
    fig.show()

def plot_key_indicators(ts, indicators):
    '''Plot key indicators as a box chart'''
    fig = go.Figure()
    fig.add_trace(go.Box(y=ts.values, name='Data', boxmean=True))
    fig.update_layout(title='Key Indicators', xaxis_title='Indicator', yaxis_title='Value', template='plotly_white')
    fig.show()

def plot_train_test(train, test, title='Train vs Test', x_title='Date', y_title='Value'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train.values, mode='lines', name='Train'))
    fig.add_trace(go.Scatter(x=test.index, y=test.values, mode='lines', name='Test'))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, template='plotly_white')
    fig.show()

def plot_forecast(test, forecast, title='Forecast vs Actual', x_title='Date', y_title='Value'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test.index, y=test.values, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=test.index, y=forecast.values, mode='lines', name='Forecast'))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, template='plotly_white')
    fig.show()