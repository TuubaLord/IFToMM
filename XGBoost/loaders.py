import pandas as pd
import numpy as np
import re
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

EVENT_PATH = "../CARE_To_Compare/Wind Farm C/datasets/"
EVENT_INFO = "../CARE_To_Compare/Wind Farm C/event_info.csv"
PCA_COMPONENTS = 170
default_noise = False
# def load_event_for_training(event_n):
#     load_event(event_n)


# def load_event_for_testing(event_n):
#     pass



def load_event(event_n, noise = default_noise, window_size = 6, undersample = False, pca=None, drop47 = True):
    X = None
    events = pd.read_csv(EVENT_INFO, sep=";")
    event_cls = None
    event_type = events[events["event_id"] == event_n]["event_label"].values[0]
    if event_type == "anomaly":
        event_cls = 1.
    else:
        event_cls = 0.
    
    event = pd.read_csv(f"{EVENT_PATH}{event_n}.csv", sep=";")
    event = event.dropna()
    y = np.where(
        event['train_test'] == "train",
        0,  event_cls)
    timestamps = event['time_stamp'].values
    
    # use all (avg|max|min|std)
    # use all (sensor|wind_speed|power|reactive_power)
    regex_pattern = 'avg'
    sensor47 = f'sensor_47_{regex_pattern}'
    sensor_pattern = re.compile(rf'^(sensor|wind_speed|power|reactive_power)_\d+_({regex_pattern})$')
    if not drop47:
        selected_columns = [col for col in event.columns if sensor_pattern.match(col)]
        X = event[selected_columns].values
    else:
    
        # 3a. Create a mask to select rows that should be KEPT (within the 48-52 range)
        keep_mask = (event[sensor47] >= 48) & \
                    (event[sensor47] <= 52)
        
        # 3b. Apply the filter to the entire DataFrame
        event_filtered = event[keep_mask]
        
        # 3c. Re-select the desired feature columns from the filtered data
        selected_columns = [col for col in event_filtered.columns if sensor_pattern.match(col)]
        
        # 3d. Extract the values into the final feature matrix X
        X = event_filtered[selected_columns].values
        y = y[keep_mask]
        timestamps = timestamps[keep_mask]
        
    if noise:
        noise_level = 0.01
        y += np.random.normal(loc=0.0, scale=noise_level, size=y.shape)
        y = np.clip(y, -1, 1)
    # pca = PCA(n_components=10, svd_solver='full')
    # X = pca.fit_transform(X)
    if window_size != 1:
        # Check if we need to truncate the data to fit the window size
        remainder = y.shape[0] % window_size
        if remainder != 0:
            y = y[:-remainder]
            X = X[:-remainder]
            timestamps = timestamps[:-remainder]
        
        # FIX: Assign the result back to 'y'
        y = y.reshape((-1, window_size)).mean(axis=1)
        
        # X is already being assigned correctly
        X = X.reshape((-1, window_size, X.shape[1])).mean(axis=1)
        
        # Timestamps: take the first time of each window
        timestamps = timestamps.reshape((-1, window_size))[:, 0]
    if undersample:
        anomaly_indices = np.where(y > 0.5)[0]
        normal_indices = np.where(y <= 0.5)[0]
        #drop last 20 % of normal indeces
        normal_indices = normal_indices[:int(len(normal_indices)*0.6)]
        ratio = len(normal_indices) // len(anomaly_indices)
        undersampled_normal_indices = normal_indices[::ratio]
        selected_indices = np.concatenate([anomaly_indices, undersampled_normal_indices])
        X = X[selected_indices]
        y = y[selected_indices]
        timestamps = timestamps[selected_indices]
    if pca:
        X = pca.transform(X)
    return X, y, timestamps


def load_events(event_ns, noise=default_noise, undersample = False, pca=True):
    """
    Loads and concatenates the feature matrix (X) and target vector (y) 
    for a list of event numbers.
    """
    all_X = []
    all_y = []
    
    print(f"Loading {len(event_ns)} events: {event_ns}")

    for n in event_ns:
        X_event, y_event, _ = load_event(n, noise=noise, undersample=undersample) 
            
        all_X.append(X_event)
        all_y.append(y_event)
        
    X_final = np.concatenate(all_X, axis=0)
    y_final = np.concatenate(all_y, axis=0)
    
    print("-" * 30)
    print(f"Load complete. Final X shape: {X_final.shape}")
    print(f"Final Y shape: {y_final.shape}")
    if pca:
        n_components_target = PCA_COMPONENTS
        pca = PCA(n_components=n_components_target)
        X_final = pca.fit_transform(X_final)

    return X_final, y_final, pca


# X, y, _ = load_event(1)
# print(X.shape, y.shape)