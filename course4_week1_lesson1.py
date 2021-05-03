import numpy as np
import matplotlib.pyplot as plt


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


def autocorrelation(time, amplitude):
    rho1 = 0.5
    rho2 = -0.1
    ar = np.random.randn(len(time) + 50)
    ar[:50] = 100
    for step in range(len(time)):
        ar[step+50] = ar[step+50] + rho1 * ar[step]
        ar[step+33] = ar[step+33] + rho2 * ar[step]
    return ar[50:] * amplitude


# show trend
time = np.arange(4*365 + 1)
pattern = trend(time, 0.1)
series = np.copy(pattern)
plot_series(time, series, "trend")

# show trend + seasonality
pattern = seasonality(time, 365, 40)
plot_series(time, pattern, "seasonality")
series += pattern
plot_series(time, series, "trend + seasonality")

# show trend + seasonality + noise
pattern = noise(time, 2)
plot_series(time, pattern, "noise")
series += pattern
plot_series(time, series, "trend + seasonality + noise")

# show trend + seasonality + noise + autocorrelation
pattern = autocorrelation(time, 5)
plot_series(time, pattern, "autocorrelation")
series += pattern
plot_series(time, series, "trend + seasonality + noise + autocorrelation")

plt.show()