from show_prediction import predict_and_plot
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
roc_curves = []
def plot_mean_roc(roc_curves, ax=None, curve_color = 'b', label = 'Case 3'):
    """
    Plots the Mean ROC curve with ±1 Standard Deviation shading.
    
    Args:
        roc_curves: List of tuples (fpr, tpr, thresholds)
        ax: (Optional) matplotlib axis to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for curve in roc_curves:
        fpr, tpr, _ = curve
        # Interpolate TPR to the common FPR grid
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        # Calculate AUC for this specific event
        aucs.append(auc(fpr, tpr))

    tprs = np.array(tprs)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    
    mean_auc = auc(mean_fpr, mean_tpr)
    if mean_auc < 0.5:
        print("Warning: Mean AUC < 0.5, inverting curve for case label:", label)
        mean_auc = 1 - mean_auc
        mean_tpr = 1 - mean_tpr[::-1]
        std_tpr = std_tpr[::-1]

    std_auc = np.std(aucs)
    print(mean_auc, std_auc)
    # Plot Mean Curve
    ax.plot(mean_fpr, mean_tpr, color=curve_color,
            label=label,
            lw=2, alpha=.8)
            # label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),

    # Plot Shading
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    if label == 'Case 3':
        ax.fill_between(mean_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.2,
                    label=r'$\pm$ 1 std. dev.')
    else:
        ax.fill_between(mean_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.2)
    # Plot Random Guess
    if label == 'Case 3':
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', label='Random Guess', alpha=.8)
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
            title="Receiver Operating Characteristic")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        
    return ax
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
        roc_curves.append(roc_curve(y_true, y_pred_proba))
        return roc_auc
    except ValueError as e:
        print(f"Error calculating ROC AUC: {e}")
        print("Ensure 'y_true' contains only binary labels (0s and 1s) and 'y_pred_proba' contains continuous probabilities.")
        return None


def print_auc_scores(y, y_pred):


    # Calculate the score
    auc_score = calculate_roc_auc(y, y_pred)

    # if auc_score is not None:
    #     print(f"True labels (y) shape: {y.shape}")
    #     print(f"Prediction scores (y_pred) shape: {y_pred.shape}")
    #     print(f"\nComputed ROC AUC Score: {auc_score:.4f}")


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


# tolppa 12: f 15, 66 h 50
train = [66]
test = 15
y, y_pred = predict_and_plot(test, train)
print_auc_scores(y, y_pred)
train = [15]
test = 66
y, y_pred = predict_and_plot(test, train)
print_auc_scores(y, y_pred)


#tolppa 35: f 31, 67 h 58, 48
train = [31]
test = 67
y, y_pred = predict_and_plot(test, train)
print_auc_scores(y, y_pred)

train = [67]
test = 31
y, y_pred = predict_and_plot(test, train)
print_auc_scores(y, y_pred)

#tolppa 52: f 28, 39 h 54, 43
train = [28]
test = 39
y, y_pred = predict_and_plot(test, train)
print_auc_scores(y, y_pred)

train = [39]
test = 28
y, y_pred = predict_and_plot(test, train)
print_auc_scores(y, y_pred)


#tolppa 53: f 35, 16, 76 h 1, 20, 60 | 35 is 8 minute standstills, cannot be detected in 1h windows
train = [76]
test = 16
y, y_pred = predict_and_plot(test, train)
print_auc_scores(y, y_pred)

train = [16]
test = 76
y, y_pred = predict_and_plot(test, train)
print_auc_scores(y, y_pred)

ax1 = plot_mean_roc(roc_curves, curve_color='r', label='Case 1')

case_n = 2
roc_curves = []
#tolppa 12
train = [30, 31, 67, 28, 39, 16, 76]
test = 15
y, y_pred = predict_and_plot(test, train, case_n=case_n)
print_auc_scores(y, y_pred)
test = 66
y, y_pred = predict_and_plot(test, train, case_n=case_n)
print_auc_scores(y, y_pred)

#tolppa 16
train = [15, 66, 31, 67, 28, 39, 16, 76]
test = 30
y, y_pred = predict_and_plot(test, train, case_n=case_n)
print_auc_scores(y, y_pred)


#tolppa 35
train = [15, 66, 30, 28, 39, 16, 76]
test = 31
y, y_pred = predict_and_plot(test, train, case_n=case_n)
print_auc_scores(y, y_pred)
test = 67
y, y_pred = predict_and_plot(test, train, case_n=case_n)
print_auc_scores(y, y_pred)

#tolppa 52
train = [15, 66, 30, 31, 67, 16, 76]
test = 28
y, y_pred = predict_and_plot(test, train, case_n=case_n)
print_auc_scores(y, y_pred)
test = 39
y, y_pred = predict_and_plot(test, train, case_n=case_n)
print_auc_scores(y, y_pred)

#tolppa 53
train = [15, 66, 30, 31, 67, 28, 39]
test = 16
y, y_pred = predict_and_plot(test, train, case_n=case_n)
print_auc_scores(y, y_pred)
test = 76
y, y_pred = predict_and_plot(test, train, case_n=case_n)
print_auc_scores(y, y_pred)

ax2 = plot_mean_roc(roc_curves, ax = ax1, curve_color='g', label='Case 2')
roc_curves = []

case_n = 3
#tolppa 12
train = [30, 31, 67, 28, 39, 16, 76]
test = 15
y, y_pred = predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=50)
print_auc_scores(y, y_pred)
test = 66
y, y_pred = predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=50)
print_auc_scores(y, y_pred)

#test = 50 ei voida testaa, tolpassa vaa 1h
#predict_and_plot(test, train, case_n=case_n)
#tolppa 16
train = [15, 66, 31, 67, 28, 39, 16, 76]
test = 30
y, y_pred = predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=65)
print_auc_scores(y, y_pred)


#tolppa 35
train = [15, 66, 30, 28, 39, 16, 76]
test = 31
y, y_pred = predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=58)
print_auc_scores(y, y_pred)

test = 67
y, y_pred = predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=58)
print_auc_scores(y, y_pred)



#tolppa 52
train = [15, 66, 30, 31, 67, 16, 76]
test = 28
y, y_pred = predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=54)
print_auc_scores(y, y_pred)

test = 39
y, y_pred = predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=54)
print_auc_scores(y, y_pred)


#tolppa 53
train = [15, 66, 30, 31, 67, 28, 39]
test = 16
y, y_pred = predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=60)
print_auc_scores(y, y_pred)

test = 76
y, y_pred = predict_and_plot(test, train, case_n=case_n, normalize_with_healthy=60)
print_auc_scores(y, y_pred)


ax3 = plot_mean_roc(roc_curves, ax = ax2, curve_color='b', label='Case 3')
plt.savefig("../results/roc_curves.png", dpi=600)

# opt_threshold, max_f1_score = find_optimal_threshold(y, y_pred)

# print(f"Optimal Threshold (Max F1): {opt_threshold:.4f}")
# print(f"Maximum F1-Score achieved: {max_f1_score:.4f}")