from show_prediction import predict_and_plot
from sklearn.metrics import roc_auc_score, precision_recall_curve
import numpy as np

def find_optimal_threshold(y_true, y_pred_proba):
    """
    Finds the optimal threshold by maximizing the F1-score.
    """
    
    # 1. Calculate Precision, Recall, and Thresholds
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Ensure arrays are the same length for F1 calculation (thresholds has one less element)
    # We remove the last precision/recall point as it corresponds to the max threshold (1.0)
    precision = precision[:-1]
    recall = recall[:-1]
    
    # 2. Compute F1-Score for every threshold
    # Suppress warnings for division by zero (happens when Precision + Recall = 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precision * recall) / (precision + recall)
    
    # 3. Find the index corresponding to the maximum F1-Score
    optimal_idx = np.argmax(f1_scores)
    
    # 4. Retrieve the optimal threshold and the maximum F1-score
    optimal_threshold = thresholds[optimal_idx]
    max_f1 = f1_scores[optimal_idx]
    
    return optimal_threshold, max_f1
def calculate_roc_auc(y_true, y_pred_proba):
    """
    Calculates the ROC AUC score.

    Args:
        y_true (np.array): Array of true binary labels (0 or 1).
        y_pred_proba (np.array): Array of probability predictions (e.g., from 0.0 to 1.0).

    Returns:
        float: The computed ROC AUC score.
    """
    
    # Check if y_pred_proba is suitable (should not be hard 0s or 1s unless you are confident)
    # The roc_auc_score function handles the necessary threshold sweeping internally.
    
    try:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        return roc_auc
    except ValueError as e:
        print(f"Error calculating ROC AUC: {e}")
        print("Ensure 'y_true' contains only binary labels (0s and 1s) and 'y_pred_proba' contains continuous probabilities.")
        return None


def print_auc_scores(y, y_pred):


    # Calculate the score
    auc_score = calculate_roc_auc(y, y_pred)

    if auc_score is not None:
        print(f"True labels (y) shape: {y.shape}")
        print(f"Prediction scores (y_pred) shape: {y_pred.shape}")
        print(f"\nComputed ROC AUC Score: {auc_score:.4f}")


# train = [30, 31, 67, 28, 39, 16, 76]
# test = 15

# y, y_pred = predict_and_plot(test, train)

# # Calculate the score
# auc_score = calculate_roc_auc(y, y_pred)

# if auc_score is not None:
#     print(f"True labels (y) shape: {y.shape}")
#     print(f"Prediction scores (y_pred) shape: {y_pred.shape}")
#     print(f"\nComputed ROC AUC Score: {auc_score:.4f}")

# opt_threshold, max_f1_score = find_optimal_threshold(y, y_pred)

# print(f"Optimal Threshold (Max F1): {opt_threshold:.4f}")
# print(f"Maximum F1-Score achieved: {max_f1_score:.4f}")

case_n = 3
train = [30, 31, 67, 28, 39, 16, 76, 50]
test = 15
y, y_pred = predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=50)
print_auc_scores(y, y_pred)
test = 66
y, y_pred = predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=50)
print_auc_scores(y, y_pred)

#test = 50 ei voida testaa, tolpassa vaa 1h
#predict_and_plot(test, train, case_n=case_n)
#tolppa 16
train = [15, 66, 31, 67, 28, 39, 16, 76, 65]
test = 30
y, y_pred = predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=65)
print_auc_scores(y, y_pred)

test = 46
y, y_pred = predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=65)
print_auc_scores(y, y_pred)

train = [15, 66, 31, 67, 28, 39, 16, 76, 46]
test = 65
y, y_pred = predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=46)
print_auc_scores(y, y_pred)

#tolppa 35
train = [15, 66, 30, 28, 39, 16, 76, 48]
test = 31
y, y_pred = predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=48)
print_auc_scores(y, y_pred)

test = 67
y, y_pred = predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=48)
print_auc_scores(y, y_pred)

test = 58
y, y_pred = predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=48)
print_auc_scores(y, y_pred)

train = [15, 66, 30, 28, 39, 16, 76, 58] 
test = 48
y, y_pred = predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=58)
print_auc_scores(y, y_pred)


#tolppa 52
train = [15, 66, 30, 31, 67, 16, 76, 43]
test = 28
y, y_pred = predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=43)
print_auc_scores(y, y_pred)

test = 39
y, y_pred = predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=43)
print_auc_scores(y, y_pred)

test = 54
y, y_pred = predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=43)
print_auc_scores(y, y_pred)

train = [15, 66, 30, 31, 67, 16, 76, 54]
test = 43
y, y_pred = predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=54)
print_auc_scores(y, y_pred)

#tolppa 53
train = [15, 66, 30, 31, 67, 28, 39, 60]
test = 16
y, y_pred = predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=60)
print_auc_scores(y, y_pred)

test = 76
y, y_pred = predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=60)
print_auc_scores(y, y_pred)

test = 20
y, y_pred = predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=60)
print_auc_scores(y, y_pred)

train = [15, 66, 30, 31, 67, 28, 39, 20]
test = 60
y, y_pred = predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=20)
print_auc_scores(y, y_pred)



# opt_threshold, max_f1_score = find_optimal_threshold(y, y_pred)

# print(f"Optimal Threshold (Max F1): {opt_threshold:.4f}")
# print(f"Maximum F1-Score achieved: {max_f1_score:.4f}")