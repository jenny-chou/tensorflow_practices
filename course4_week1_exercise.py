import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


def plot_series(time, series, title=""):
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
    for time in range(len(series)-window_size):
        window = series[time:time+window_size]
        forecast.append(window.mean())
    return np.array(forecast)


time = np.arange(4*365+1, dtype='float')
series = 10 + trend(time, 0.01) + seasonality(time, 365, 40) + noise(time, 2)

split_time = 1000 # YOUR CODE HERE
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
plt.figure(figsize=(10, 6))
plt.plot(time_train, x_train)
plt.plot(time_valid, x_valid)
plt.title("train and test set")

naive_forecast = series[split_time-1:-1] #YOUR CODE HERE]

plt.figure(figsize=(10, 6))
plt.plot(time_valid, x_valid)
plt.plot(time_valid, naive_forecast)
plt.title("naive forecast")

plt.figure(figsize=(10, 6))
plt.plot(time_valid[0:150], x_valid[0:150]) # YOUR CODE HERE)
plt.plot(time_valid[1:151], naive_forecast[1:151]) # YOUR CODE HERE)
plt.title("naive forecast zoom in")

print(keras.metrics.mean_squared_error(x_valid[1:], naive_forecast[1:]).numpy())  # YOUR CODE HERE)
print(keras.metrics.mean_absolute_error(x_valid[1:], naive_forecast[1:]).numpy())  # YOUR CODE HERE)

def moving_average_forecast(series, window_size):
  """Forecasts the mean of the last few values.
     If window_size=1, then this is equivalent to naive forecast"""
  # YOUR CODE HERE
  avg_series = []
  for i in range(len(series[window_size:])):
      avg_series.append(np.mean(series[i:i+window_size]))
  return np.array(avg_series)

moving_avg = moving_average_forecast(series, 50)[split_time-50:]  # YOUR CODE HERE)[# YOUR CODE HERE]

plt.figure(figsize=(10, 6))
plt.plot(time_valid, x_valid)
plt.plot(time_valid, moving_avg)
plt.title("moving avg")

print(keras.metrics.mean_squared_error(x_valid, moving_avg).numpy())  # YOUR CODE HERE)
print(keras.metrics.mean_absolute_error(x_valid, moving_avg).numpy())  # YOUR CODE HERE)

diff_series = (series[365:] - series[0: len(series[365:])])  # YOUR CODE HERE)
diff_time = time[365:]  # YOUR CODE HERE

plt.figure(figsize=(10, 6))
plt.plot(diff_time, diff_series)
plt.title("difference series")

diff_moving_avg = moving_average_forecast(diff_series, 50)  # YOUR CODE HERE

plt.figure(figsize=(10, 6))
plt.plot(time_valid, diff_series[split_time-365:])# YOUR CODE HERE)
plt.plot(time_valid, diff_moving_avg[split_time-365-50:])# YOUR CODE HERE)
plt.title("difference moving avg")

diff_moving_avg_plus_past = diff_moving_avg + series[50:-365]  # YOUR CODE HERE

plt.figure(figsize=(10, 6))
plt.plot(time_valid, x_valid)  # YOUR CODE HERE)
plt.plot(time_valid, diff_moving_avg_plus_past[split_time-365-50:])  # YOUR CODE HERE)
plt.title("difference moving avg plus past")

print(keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_past[split_time-365-50:]).numpy())  # YOUR CODE HERE)
print(keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_past[split_time-365-50:]).numpy())  # YOUR CODE HERE)

diff_moving_avg_plus_smooth_past = diff_moving_avg[10:] + moving_average_forecast(series[50:-365], 10)# YOUR CODE HERE

plt.figure(figsize=(10, 6))
plt.plot(time_valid, x_valid)  # YOUR CODE HERE)
plt.plot(time_valid, diff_moving_avg_plus_smooth_past[split_time-365-50-10:])  # YOUR CODE HERE)
plt.title("difference moving avg plus smooth past")
plt.show()

print(keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_smooth_past[split_time-365-50-10:]))  # YOUR CODE HERE)
print(keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_smooth_past[split_time-365-50-10:]))  # YOUR CODE HERE)