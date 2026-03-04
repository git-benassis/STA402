import yfinance as yf
import pandas as pd

def download_spy_data(interval="1d"):
    spy = yf.download("SPY", interval=interval, period="max")
    save_path ="SPY_daily.csv"
    spy.to_csv(save_path)
    return spy

def load_spy_data(file_path="SPY_daily.csv"):
    spy = pd.read_csv(file_path, index_col=0, parse_dates=True)
    print(spy.head())
    return spy

def keep_volume(df:pd.DataFrame):
    return df['Volume'].squeeze()

def train_test_split(ts:pd.Series, test_size:float=0.2, specific_date=[]):
    if specific_date:
        train = ts[specific_date[0]:specific_date[1]]
        test = ts[specific_date[1]:]
    else:
        split_index = int(len(ts) * (1 - test_size))
        train = ts.iloc[:split_index]
        test = ts.iloc[split_index:]
    return train, test