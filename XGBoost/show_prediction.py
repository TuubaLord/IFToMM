from loaders import load_event, load_events
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from helpers import get_model
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
import math

mpl.rcParams['figure.dpi'] = 600

def smooth(scalars: list[float], weight: float) -> list[float]:
    """
    EMA implementation according to
    https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699
    """
    last = 0
    smoothed = []
    num_acc = 0
    for next_val in scalars:
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        # de-bias
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - math.pow(weight, num_acc)
        smoothed_val = last / debias_weight
        smoothed.append(smoothed_val)

    return smoothed

def plot_regression_results(timestamps, y_true, y_predicted, event_id, rolling_average=1, smoothen=False, display = False, test=None, train=None):
    """
    Plots true vs predicted values with an optional rolling average for the prediction.
    
    Args:
        rolling_average (int): Window size for the moving average. 
                               1 = Raw data (no smoothing).
                               >1 = Plots smoothed line.
    """
    plt.figure(figsize=(12, 6))

    # Convert timestamps to datetime objects
    dates = pd.to_datetime(timestamps)

    # Plot True Label
    plt.plot(dates, y_true, label='True Score', color='blue', linewidth=2, alpha=0.6)

    # --- ROLLING AVERAGE LOGIC ---
    if rolling_average > 1:
        # Calculate rolling mean
        # We convert y_predicted to a Series to use the powerful rolling() method
        y_smooth = pd.Series(y_predicted).rolling(window=rolling_average, min_periods=1).median()

        if smoothen:
            y_smooth = smooth(y_smooth, 0.6)
        
        # Plot the smoothed line
        plt.plot(dates, y_smooth, label=f'Prediction ({rolling_average}-step Moving Avg)', 
                 color='red', linestyle='--', marker='.', linewidth=2)
        
        # Optional: Plot the raw noisy prediction faintly in the background
        #plt.plot(dates, y_predicted, color='red', linestyle='-', linewidth=0.5, alpha=0.2)
        
    else:
        # Standard Plot (No smoothing)
        if smoothen:
            y_predicted = smooth(y_predicted, 0.2)
        plt.plot(dates, y_predicted, label='Prediction', color='red', linestyle='--', marker='.', linewidth=1.5)

    # --- AXIS FORMATTING ---

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # Labels and Titles
    plt.title(f"Event {event_id}: True vs Predicted Healthiness")
    plt.xlabel("Date")
    plt.ylabel("Healthiness (0=Normal, 1=Anomaly)")
    plt.ylim(-.05, 1.05)
    
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    if display:
        plt.show()
    else:
        plt.savefig(f"results/case1/test_{test}_train_{train}.png")


def predict_and_plot(test, train):
    test_event = test# faulty 15 67 healthy 50
    #train_events = [52, 21, 2, 23, 87, 74, 86, 82]
    train_events = train
    use_test = False
    #train_events = []
    train_noise = True
    test_noise = False
    undersample_train = True# Example usage (place this line after running your model fit/predict steps):
    # plot_regression_results(timestamp, y_test, y_pred)
    # X_test, y_test, timestamp = load_event(test_event, noise=test_noise)
    X_train, y_train, pca = None, None, None
    if train_events != []:
        X_train, y_train, pca = load_events(train_events, noise=train_noise, undersample=undersample_train)


    X, y, timestamp = load_event(test_event, undersample=False, pca=pca)
    if use_test: 
        if train_events == []:
            X_train = X[0:X.shape[0]//2]
            y_train = y[0:X.shape[0]//2] 
        else:
            X_train = np.concatenate([X_train, X[0:X.shape[0]//2]])
            y_train = np.concatenate([y_train, y[0:X.shape[0]//2]])
    # X_test, y_test = X[X.shape[0]//2:], y[X.shape[0]//2:]
    # timestamp = timestamp[X.shape[0]//2:]
    X_test, y_test, timestamp = X, y, timestamp
    model = get_model(type=3)


    model.fit(X_train, y_train)
    #model.fit(X_test, y_test)

    # y_pred_raw = model.predict(X_test)
    y_pred = model.predict(X_test)
    y_pred = y_pred - y_pred[0:24 * 30].mean()
    y_pred = np.maximum(y_pred, 0)
    y_pred = np.minimum(y_pred, 1)
    th = 0.8


    # 2. Apply Smoothing (Crucial for Autoencoders to reduce noise)
    # Window 144 = 24 hours (assuming 10 min intervals)
    #y_pred_smooth = pd.Series(y_pred_raw).rolling(window=144, min_periods=1).mean()

    # 3. Plot
    plot_regression_results(timestamp, y_test, y_pred, test_event, rolling_average=24, test=test, train=train)




# tolppa 12: f 15, 66 h 50
train = [66]
test = 15
predict_and_plot(test, train)
train = [15]
test = 66
predict_and_plot(test, train)
train = [15]
test = 50
predict_and_plot(test, train)

#tolppa 16: f 79, 30 h 46, 65 | 79 is communication failure -> 30 and 79 comparisons not usable
train = [79]
test = 30
predict_and_plot(test, train)
train = [30]
test = 79
predict_and_plot(test, train)
train = [30]
test = 46
predict_and_plot(test, train)
train = [30]
test = 65
predict_and_plot(test, train)

#tolppa 35: f 31, 67 h 58, 48
train = [31]
test = 67
predict_and_plot(test, train)
train = [67]
test = 31
predict_and_plot(test, train)
train = [67]
test = 58
predict_and_plot(test, train)
train = [67]
test = 48
predict_and_plot(test, train)

#tolppa 52: f 28, 39 h 54, 43
train = [28]
test = 39
predict_and_plot(test, train)
train = [39]
test = 28
predict_and_plot(test, train)
train = [28]
test = 54
predict_and_plot(test, train)
train = [28]
test = 43
predict_and_plot(test, train)

#tolppa 53: f 35, 16, 76 h 1, 20, 60 | 35 is 8 minute standstills, cannot be detected in 1h windows
train = [76]
test = 16
predict_and_plot(test, train)
train = [16]
test = 35
predict_and_plot(test, train)
train = [16]
test = 76
predict_and_plot(test, train)
train = [16]
test = 1
predict_and_plot(test, train)
train = [16]
test = 20
predict_and_plot(test, train)
train = [16]
test = 60
predict_and_plot(test, train)