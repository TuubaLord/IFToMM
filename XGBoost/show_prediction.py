from loaders import load_event, load_events
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from helpers import get_model, sigmoid
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
import math
from sklearn.metrics import confusion_matrix
import json
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import IsolationForest

mpl.rcParams['figure.dpi'] = 600
cms = {}
roc_aucs = {}
def min_max_magic(y_pred):
    # y_pred = y_pred - y_pred[0:24 * 30].mean()
    #y_pred = np.maximum(y_pred, 0)
    #y_pred = np.minimum(y_pred, 1)

    return sigmoid(y_pred)

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

def plot_regression_results(timestamps, y_true, y_predicted, event_id, rolling_average=1, smoothen=False, display = False, test=None, train=None, dict = cms, threshold=0.23, case_n=1):
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
    plt.grid(True, alpha=0.5)
    plt.hlines(y=threshold, color='grey', linestyle='--', label='Anomaly threshold', xmin=dates.min(), xmax=dates.max())
    plt.legend()
    plt.tight_layout()    
    if display:
        plt.show()
    else:
        plt.savefig(f"results/case{case_n}/test_{test}_train_{train}.png")
        plt.close()

def predict_and_plot(test, train, case_n=1, model_type = 3, undersample_test = False, normalize_with_healthy = False):
    th = 0.8

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
    model = get_model(type=model_type)


    model.fit(X_train, y_train)
    #model.fit(X_test, y_test)
    y_pred = None
    if normalize_with_healthy:
        healthy_X, _, _ = load_event(normalize_with_healthy, pca=pca, remove_middle=False, window_size=6, undersample=False)

        healthy_leaves = model.apply(healthy_X)

        leaf_encoder = OneHotEncoder(categories='auto', sparse_output=True, handle_unknown='ignore')
        healthy_leaves_emb = leaf_encoder.fit_transform(healthy_leaves)

        transfer_detector = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        transfer_detector.fit(healthy_leaves_emb)
            
    if normalize_with_healthy:
        test_leaves = model.apply(X_test)
        
        test_leaves_emb = leaf_encoder.transform(test_leaves)
        
        iso_preds = transfer_detector.predict(test_leaves_emb)
        raw_preds = transfer_detector.score_samples(test_leaves_emb)
        print(raw_preds.min(), raw_preds.max(), raw_preds.mean(), raw_preds.std())
        y_pred = -1 * raw_preds
        th = -raw_preds.mean() + 0.01
        
    else:
        y_pred = model.predict(X_test)
        y_pred = min_max_magic(y_pred)

    # y_pred = model.predict(X_test)
    # y_pred = min_max_magic(y_pred)
    plot_regression_results(timestamp, y_test, y_pred, test_event, rolling_average=24, test=test, train=train, threshold=th, case_n=case_n)

    X_for_cm, y_for_cm, _ = load_event(test_event, pca=pca, remove_middle=(not undersample_test), window_size=6, undersample = undersample_test)
    y_pred_for_cm = None
    if normalize_with_healthy:
        test_leaves = model.apply(X_for_cm)
        
        test_leaves_emb = leaf_encoder.transform(test_leaves)
        
        iso_preds = transfer_detector.predict(test_leaves_emb)
        raw_preds = transfer_detector.score_samples(test_leaves_emb)
        print(raw_preds.min(), raw_preds.max(), raw_preds.mean(), raw_preds.std())
        y_pred_for_cm = -1 * raw_preds        
        
        
    else:
        y_pred_for_cm = model.predict(X_for_cm)
        y_pred_for_cm = min_max_magic(y_pred_for_cm)
    
    #y_pred_for_cm = min_max_magic(y_pred_for_cm)
    
    
    # y_pred_class = (y_pred_for_cm >= th).astype(int)

    # plt.plot(y_for_cm)
    # plt.plot(y_pred_class)
    # plt.show()

    #y_for_cm = y_for_cm[24*30:]
    #y_pred_for_cm = y_pred_for_cm[24*30:]

    y_pred_for_cm = pd.Series(y_pred_for_cm).rolling(window=24, min_periods=1).median().to_numpy()
    y_pred_class = (y_pred_for_cm >= th).astype(int)

    cm = confusion_matrix(y_for_cm.astype(int), y_pred_class, labels=[0,1])


    cms[f"test_{test}_train_{train}"] = cm.tolist()
    return y_for_cm.astype(int), y_pred_for_cm

    # 2. Apply Smoothing (Crucial for Autoencoders to reduce noise)
    # Window 144 = 24 hours (assuming 10 min intervals)
    #y_pred_smooth = pd.Series(y_pred_raw).rolling(window=144, min_periods=1).mean()

    # 3. Plot

case_n = 3333
if case_n == 1:

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
    # train = [79]
    # test = 30
    # predict_and_plot(test, train)
    # train = [30]
    # test = 79
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
    #train = [16]
    #test = 35
    #predict_and_plot(test, train)
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
    with open("results/case1/cms.json", "w") as f:
        json.dump(cms, f, indent=4)

if case_n == 2:
    # train with other faulty, test with different turbine

    #tolppa 12
    train = [30, 31, 67, 28, 39, 16, 76]
    test = 15
    predict_and_plot(test, train, case_n=case_n)
    test = 66
    predict_and_plot(test, train, case_n=case_n)
    test = 50
    predict_and_plot(test, train, case_n=case_n)
    #tolppa 16
    train = [15, 66, 31, 67, 28, 39, 16, 76]
    test = 30
    predict_and_plot(test, train, case_n=case_n)
    test = 79
    predict_and_plot(test, train, case_n=case_n)
    test = 46
    predict_and_plot(test, train, case_n=case_n)
    test = 65
    predict_and_plot(test, train, case_n=case_n)
    #tolppa 35
    train = [15, 66, 30, 28, 39, 16, 76]
    test = 31
    predict_and_plot(test, train, case_n=case_n)
    test = 67
    predict_and_plot(test, train, case_n=case_n)
    test = 58
    predict_and_plot(test, train, case_n=case_n)
    test = 48
    predict_and_plot(test, train, case_n=case_n)
    #tolppa 52
    train = [15, 66, 30, 31, 67, 16, 76]
    test = 28
    predict_and_plot(test, train, case_n=case_n)
    test = 39
    predict_and_plot(test, train, case_n=case_n)
    test = 54
    predict_and_plot(test, train, case_n=case_n)
    test = 43
    predict_and_plot(test, train, case_n=case_n)
    #tolppa 53
    train = [15, 66, 30, 31, 67, 28, 39]
    test = 16
    predict_and_plot(test, train, case_n=case_n)
    test = 76
    predict_and_plot(test, train, case_n=case_n)
    test = 35
    predict_and_plot(test, train, case_n=case_n)
    test = 20
    predict_and_plot(test, train, case_n=case_n)
    test = 60
    predict_and_plot(test, train, case_n=case_n)
    with open("results/case2/cms.json", "w") as f:
        json.dump(cms, f, indent=4)



# normalize wiith healthy from tested tolppa
if case_n == 3:
# train with other faulty, test with different turbine
#tolppa 12
    train = [30, 31, 67, 28, 39, 16, 76, 50]
    test = 15
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=50)
    test = 66
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=50)
    #test = 50 ei voida testaa, tolpassa vaa 1h
    #predict_and_plot(test, train, case_n=case_n)
    #tolppa 16
    train = [15, 66, 31, 67, 28, 39, 16, 76, 65]
    test = 30
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=65)
    test = 46
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=65)
    train = [15, 66, 31, 67, 28, 39, 16, 76, 46]
    test = 65
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=46)
    #tolppa 35
    train = [15, 66, 30, 28, 39, 16, 76, 48]
    test = 31
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=48)
    test = 67
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=48)
    test = 58
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=48)
    train = [15, 66, 30, 28, 39, 16, 76, 58] 
    test = 48
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=58)
    #tolppa 52
    train = [15, 66, 30, 31, 67, 16, 76, 43]
    test = 28
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=43)
    test = 39
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=43)
    test = 54
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=43)
    train = [15, 66, 30, 31, 67, 16, 76, 54]
    test = 43
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=54)
    #tolppa 53
    train = [15, 66, 30, 31, 67, 28, 39, 60]
    test = 16
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=60)
    test = 76
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=60)
    test = 20
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=60)
    train = [15, 66, 30, 31, 67, 28, 39, 20]
    test = 60
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=20)
    with open(f"results/case{case_n}/cms.json", "w") as f:
        json.dump(cms, f, indent=4)

# Use faults from same windmill and different windmills.
if case_n == 4:

    # tolppa 12: f 15, 66 h 50
    train = [66, 30, 31, 67, 28, 39, 16, 76, 50]
    test = 15
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=50)
    train = [15, 30, 31, 67, 28, 39, 16, 76, 50]
    test = 66
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=50)
    train = [15, 30, 31, 67, 28, 39, 16, 76, 50]
    # test = 50
    # predict_and_plot(test, train, case_n=case_n)

    #tolppa 16: f 79, 30 h 46, 65 | 79 is communication failure -> 30 and 79 comparisons not usable
    # train = [79]
    # test = 30
    # predict_and_plot(test, train)
    # train = [30]
    # test = 79
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=65)
    train = [15, 30, 31, 67, 28, 39, 16, 76, 50]
    test = 46
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=65)
    train = [15, 30, 31, 67, 28, 39, 16, 76, 50]
    test = 65
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=46)

    #tolppa 35: f 31, 67 h 58, 48
    train = [15, 30, 31, 28, 39, 16, 76, 50]
    test = 67
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=48)
    train = [15, 30, 67, 28, 39, 16, 76, 50]
    test = 31
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=48)
    test = 58
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=48)
    train = [15, 30, 67, 28, 39, 16, 76, 50]
    test = 48
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=58)

    #tolppa 52: f 28, 39 h 54, 43
    train = [15, 30, 31, 67, 28, 16, 76, 50]
    test = 39
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=43)
    train = [15, 30, 31, 67, 39, 16, 76, 50]
    test = 28
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=43)
    test = 54
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=43)
    test = 43
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=54)

    #tolppa 53: f 35, 16, 76 h 1, 20, 60 | 35 is 8 minute standstills, cannot be detected in 1h windows
    train = [15, 30, 31, 67, 28, 39, 76, 50]
    test = 16
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=60)
    #train = [16]
    #test = 35
    #predict_and_plot(test, train)
    train = [15, 30, 31, 67, 28, 39, 16, 50]
    test = 76
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=60)
    test = 1
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=60)
    test = 20
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=60)
    test = 60
    predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=20)
    with open(f"results/case{case_n}/cms.json", "w") as f:
        json.dump(cms, f, indent=4)
