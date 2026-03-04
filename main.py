import download, treatment, prediction, plot
import pandas as pd

spy_500 = pd.DataFrame(download.load_spy_data()["2005":])
volume = download.keep_volume(spy_500)

# look at our data
plot.plot_key_indicators(volume, treatment.key_indicators(volume))
plot.plot_data(volume.index, volume.values, mode="lines", name="Volume", title="SPY Volume Over Time", x_title="Date", y_title="Volume")

# look at extrordinary values
anomalies = treatment.anomalies(volume)
plot.plot_data(anomalies.index, anomalies.values, mode="markers", name="Anomalies", title="Volume Anomalies", x_title="Date", y_title="Volume")

# look at distribution
plot.plot_distribution(volume, title="Volume Distribution", x_title="Volume", y_title="Frequency")

# look at acf and pacf
acf, pacf = treatment.acf_pacf(volume)
plot.plot_acf_pacf(acf, pacf)

# treat data
log_volume = treatment.log_volume(volume)
plot.plot_key_indicators(log_volume, treatment.key_indicators(log_volume))
plot.plot_data(log_volume.index, log_volume.values, mode="lines", name="Log Volume", title="Log of SPY Volume Over Time", x_title="Date", y_title="Log Volume")
plot.plot_distribution(log_volume, title="Log Volume Distribution", x_title="Log Volume", y_title="Frequency")

# look at seasonality
month_partition = treatment.partition(log_volume, freq="month")
week_partition = treatment.partition(log_volume, freq="week")
dayly_partition = treatment.partition(log_volume, freq="dayofweek")
day_partition = treatment.partition(log_volume, freq="day")
yearly_partition = treatment.partition(log_volume, freq="dayofyear")

plot.plot_data(month_partition.index, month_partition.values, mode="lines+markers", name="Monthly Average Volume", title="Average Volume by Month", x_title="Month", y_title="Average Volume")
plot.plot_data(week_partition.index, week_partition.values, mode="lines+markers", name="Weekly Average Volume", title="Average Volume by Week", x_title="Week", y_title="Average Volume")
plot.plot_data(dayly_partition.index, dayly_partition.values, mode="lines+markers", name="Daily Average Volume", title="Average Volume by Day of Week", x_title="Day of Week", y_title="Average Volume")
plot.plot_data(day_partition.index, day_partition.values, mode="lines+markers", name="Daily Average Volume", title="Average Volume by Day of Month", x_title="Day of Month", y_title="Average Volume")
plot.plot_data(yearly_partition.index, yearly_partition.values, mode="lines+markers", name="Yearly Average Volume", title="Average Volume by Day of Year", x_title="Day of Year", y_title="Average Volume")

# look at stationarity

log_acf, log_pacf = treatment.acf_pacf(log_volume)
plot.plot_acf_pacf(log_acf, log_pacf)

# treat for stationarity
diff_volume = treatment.difference(log_volume)
plot.plot_data(diff_volume.index, diff_volume.values, mode="lines", name="Differenced Log Volume", title="Differenced Log of SPY Volume Over Time", x_title="Date", y_title="Differenced Log Volume")
plot.plot_distribution(diff_volume, title="Differenced Log Volume Distribution", x_title="Differenced Log Volume", y_title="Frequency")

diff_acf, diff_pacf = treatment.acf_pacf(diff_volume)
plot.plot_acf_pacf(diff_acf, diff_pacf)
    
# look at seasonality
month_partition = treatment.partition(diff_volume, freq="month")
week_partition = treatment.partition(diff_volume, freq="week")
dayly_partition = treatment.partition(diff_volume, freq="dayofweek")
day_partition = treatment.partition(diff_volume, freq="day")
yearly_partition = treatment.partition(diff_volume, freq="dayofyear")

plot.plot_data(month_partition.index, month_partition.values, mode="lines+markers", name="Monthly Average Differenced Log Volume", title="Average Differenced Log Volume by Month", x_title="Month", y_title="Average Differenced Log Volume")
plot.plot_data(week_partition.index, week_partition.values, mode="lines+markers", name="Weekly Average Differenced Log Volume", title="Average Differenced Log Volume by Week", x_title="Week", y_title="Average Differenced Log Volume")
plot.plot_data(dayly_partition.index, dayly_partition.values, mode="lines+markers", name="Daily Average Differenced Log Volume", title="Average Differenced Log Volume by Day of Week", x_title="Day of Week", y_title="Average Differenced Log Volume")
plot.plot_data(day_partition.index, day_partition.values, mode="lines+markers", name="Daily Average Differenced Log Volume", title="Average Differenced Log Volume by Day of Month", x_title="Day of Month", y_title="Average Differenced Log Volume")
plot.plot_data(yearly_partition.index, yearly_partition.values, mode="lines+markers", name="Yearly Average Differenced Log Volume", title="Average Differenced Log Volume by Day of Year", x_title="Day of Year", y_title="Average Differenced Log Volume")

#try to predict
train_log, test_log = download.train_test_split(log_volume, specific_date=["2021", "2025"])
plot.plot_train_test(train_log, test_log, title="Train vs Test Log Volume", x_title="Date", y_title="Log Volume")

arma_in_sample_forecast = prediction.arima_forecast(train_log, train_log, order=(1,1,1))
plot.plot_forecast(train_log, arma_in_sample_forecast, title="In-Sample Forecast vs Train Log Volume", x_title="Date", y_title="Log Volume")

sarimax_in_sample_forecast = prediction.sarimax_forecast(train_log, train_log, order=(1,1,1), seasonal_order=(1,0,1,5))
plot.plot_forecast(train_log, sarimax_in_sample_forecast, title="In-Sample SARIMAX Forecast vs Train Log Volume", x_title="Date", y_title="Log Volume")

arma_forecast = prediction.arima_forecast(train_log, test_log, order=(1,1,1))
plot.plot_forecast(test_log, arma_forecast, title="Forecast vs Test Log Volume", x_title="Date", y_title="Log Volume")

sarimax_forecast = prediction.sarimax_forecast(train_log, test_log, order=(1,1,1), seasonal_order=(1,0,1,5))
plot.plot_forecast(test_log, sarimax_forecast, title="SARIMAX Forecast vs Test Log Volume", x_title="Date", y_title="Log Volume")

# Evaluate
from sklearn.metrics import mean_squared_error
import numpy as np

y_pred = pd.Series(prediction.prophet_forecast(train_log, test_log), index=test_log.index)
rmse = np.sqrt(mean_squared_error(test_log, y_pred))
print("Prophet RMSE:", rmse)
plot.plot_forecast(test_log, y_pred, title="Prophet Forecast vs Test Log Volume", x_title="Date", y_title="Log Volume")

# Try with exogenous variables
exogenous_variables = treatment.compute_exogenous_variables(spy_500)
exog_train = exogenous_variables.loc[train_log.index]
exog_test = exogenous_variables.loc[test_log.index]
print(exog_train.head())
sarimax_exog_forecast = prediction.exogenous_forecast(train_log, test_log, exog_train, exog_test, order=(1,1,1))
rmse_exog = np.sqrt(mean_squared_error(test_log, sarimax_exog_forecast))
print("SARIMAX with Exogenous Variables RMSE:", rmse_exog)
plot.plot_forecast(test_log, sarimax_exog_forecast, title="SARIMAX with Exogenous Variables Forecast vs Test Log Volume", x_title="Date", y_title="Log Volume")