import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def plot_series(time, series, title):
    plt.figure(figsize=(10, 6))
    plt.plot(time, series)
    plt.xlabel("time")
    plt.ylabel("value")
    plt.grid(True)
    plt.title(title)


def trend(time, slope=0):
    return slope * time


def seasonality(time, period, amplitude=1):
    season_time = (time % period) / period  # divided by period to have range in 0~1
    season_pattern = np.where(season_time < 0.4, np.cos(season_time*2*np.pi), 1/np.exp(3*season_time))
    return amplitude * season_pattern


def noise(time, noise_level=1):
    return np.random.randn(len(time)) * noise_level


def moving_average_forecast(series, window_size):
    forecast = []
    # originally using sliding window method, there can be len(series)-window_size+1 windows from sliding through series
    # but since it's a forecasting problem, leave the last (number of timestamps we're forecasting) data points out
    # thus len(series) is reduced to len(series)-(number of timestamps we're forecasting)
    # in this case we're forecasting value at future 1 timestamp
    # thus number of windows is len(series)-1-window_size+1 = len(series)-window_size
    for time in range(len(series)-window_size):
        window = series[time:time+window_size]
        forecast.append(window.mean())  # consider this a trailing window average (not centered window average)
    return np.array(forecast)


time = np.arange(4*365+1, dtype='float')
series = 10 + trend(time, 0.1) + seasonality(time, 365, 40) + noise(time, 5)
plot_series(time, series, "baseline + trend + seasonality + noise")

split_time = 1000
train_time = time[:split_time]
test_time = time[split_time:]
train_series = series[:split_time]
test_series = series[split_time:]

plt.figure()
plt.plot(train_time, train_series)
plt.plot(test_time, test_series)
plt.title("training and testing series")

# naive forecasting: forecast for day i is time_series[i-1]
naive_forecast = series[split_time-1 : -1]
print(np.sqrt(keras.metrics.mean_squared_error(test_series, naive_forecast).numpy()))
# 7.6135162970214125
print(keras.metrics.mean_absolute_error(test_series, naive_forecast).numpy())
# 5.925858837427296

plt.figure()
plt.plot(test_time, test_series)
plt.plot(test_time, naive_forecast)
plt.title("naive forecast")

plt.figure()
plt.plot(test_time[0:150], test_series[0:150])
plt.plot(test_time[1:151], naive_forecast[1:151])
plt.title("naive forecast zoom in")

# moving average forecasting
moving_avg = moving_average_forecast(series, 30)[split_time - 30:]
print("time length:", len(time[split_time-30:]), "moving average length:", len(moving_avg))
# time length: 491
# moving average length: 461
print(np.sqrt(keras.metrics.mean_squared_error(test_series, moving_avg).numpy()))
# 10.67105414940776
print(keras.metrics.mean_absolute_error(test_series, moving_avg).numpy())
# 7.354138203400175

plt.figure()
plt.plot(test_time, test_series)
plt.plot(test_time, moving_avg)
plt.title("moving average forecasting")

# difference in time - value at current time - value at last year this time = removes trend and seasonality, leaves baseline
difference_series = series[365:] - series[0:-365]
difference_time = time[365:]

# average difference to denoise
difference_series_moving_avg = moving_average_forecast(difference_series, 50)
print(len(difference_series), len(difference_series_moving_avg))
plt.figure()
plt.plot(difference_time[50:], difference_series[50:], label="difference series")
plt.plot(difference_time[50:], difference_series_moving_avg, label="difference series moving avg")
plt.title("difference series vs difference series moving average")
plt.legend()

# on top of past series, add back the denoised differences
difference_series_moving_avg_plus_past = np.copy(series[0:-365])
difference_series_moving_avg_plus_past[50:] += difference_series_moving_avg
print(np.sqrt(keras.metrics.mean_squared_error(test_series, difference_series_moving_avg_plus_past[-len(test_series):]).numpy()))
# 7.291283868222882
print(keras.metrics.mean_absolute_error(test_series, difference_series_moving_avg_plus_past[-len(test_series):]).numpy())
# 5.791110170902592

# on top of smoothed past, add back denoised differences
difference_series_moving_avg_plus_smooth_past = moving_average_forecast(series[0:-365], 10)
difference_series_moving_avg_plus_smooth_past[50-10:] += difference_series_moving_avg
print(np.sqrt(keras.metrics.mean_squared_error(test_series, difference_series_moving_avg_plus_smooth_past[-len(test_series):]).numpy()))
# 7.939808064426621
print(keras.metrics.mean_absolute_error(test_series, difference_series_moving_avg_plus_smooth_past[-len(test_series):]).numpy())
# 5.359489489272662
print("series length:", len(series),
      "diff series length:", len(difference_series),
      "moving avg length:", len(difference_series_moving_avg),
      "plus past length:", len(difference_series_moving_avg_plus_past),
      "plus smooth past length:", len(difference_series_moving_avg_plus_smooth_past))
# series length: 1461
# diff series length: 1096
# moving avg length: 1046
# plus past length: 1096
# plus smooth past length: 1086

plt.figure()
plt.plot(test_time, test_series, label="test series")
plt.plot(test_time, difference_series_moving_avg_plus_past[-len(test_series):], label="denoised difference series")
plt.plot(test_time, difference_series_moving_avg_plus_smooth_past[-len(test_series):], label="denoised & smooth difference series")
plt.title("denoised difference series vs denoised & smooth difference series")
plt.legend()

plt.show()