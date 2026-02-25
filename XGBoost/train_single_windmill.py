from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from ipywidgets import interact, Dropdown


def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    """Train an XGBoost model and evaluate its performance."""
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Return the trained model
    return model


def plot_training_testing_data(train_data, test_data):
    """Visualize which data is used for training and testing."""
    plt.figure(figsize=(12, 6))

    # Plot training data
    plt.scatter(train_data['time_stamp'], [0] * len(train_data), color='blue', label='Training Data', alpha=0.5)

    # Plot testing data
    plt.scatter(test_data['time_stamp'], [1] * len(test_data), color='red', label='Testing Data', alpha=0.5)

    plt.title("Training and Testing Data Distribution")
    plt.xlabel("Timestamp")
    plt.yticks([0, 1], ['Training', 'Testing'])
    plt.legend()
    plt.grid()
    plt.show()


def compare_labels(event_id, events, data, model, pre_event_window, sequence_length, time_step):
    """Compare true and predicted labels for a specific event."""
    # Filter the test data for the selected event
    event = events[events['event_id'] == event_id]
    if event.empty:
        print(f"No data found for event ID {event_id}")
        return

    # Extract the data leading up to the event
    event_data = pd.DataFrame()
    for _, row in event.iterrows():
        start_time = row['event_start'] - pre_event_window
        end_time = row['event_end']
        event_data = pd.concat([event_data, data[(data['time_stamp'] >= start_time) & (data['time_stamp'] <= end_time)]])

    # Ensure the data is sorted by timestamp
    event_data = event_data.sort_values(by='time_stamp').reset_index(drop=True)

    # Create sequences for the event data
    X_event, y_event, event_timestamps = create_sequences(event_data, sequence_length, time_step)

    # Predict labels for the event data
    y_pred_event = model.predict(X_event)

    # Plot the true and predicted labels
    plt.figure(figsize=(12, 6))
    plt.plot(event_timestamps, y_event, label='True Labels', color='blue', marker='o', linestyle='-')
    plt.plot(event_timestamps, y_pred_event, label='Predicted Labels', color='red', marker='x', linestyle='--')
    plt.title(f"True vs Predicted Labels for Event ID {event_id}")
    plt.xlabel("Timestamp")
    plt.ylabel("Label")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.show()


def interactive_comparison(events, data, model, pre_event_window, sequence_length, time_step, test_event_ids):
    """Create an interactive dropdown to select an event ID for comparison."""
    interact(
        lambda event_id: compare_labels(event_id, events, data, model, pre_event_window, sequence_length, time_step),
        event_id=Dropdown(options=test_event_ids, description='Select Event ID:')
    )