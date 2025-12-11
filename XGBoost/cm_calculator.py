import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def analyze_results(filename='results.json'):
    # 1. Load Data
    try:
        with open(filename, 'r') as f:
            results_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return

    # 2. Aggregation
    joint_cm = np.zeros((2, 2), dtype=int)
    for key, cm in results_data.items():
        joint_cm += np.array(cm)

    # Structure: [[TN, FP], [FN, TP]]
    TN, FP = joint_cm[0]
    FN, TP = joint_cm[1]
    
    # Calculate Totals per Class (Rows)
    total_negatives = TN + FP
    total_positives = FN + TP

    # 3. Compute Metrics
    # Recall for Faulty (Class 1)
    recall_faulty = TP / total_positives if total_positives > 0 else 0.0
    
    # Recall for Healthy (Class 0) a.k.a Specificity
    recall_healthy = TN / total_negatives if total_negatives > 0 else 0.0

    # Unweighted Accuracy (Balanced Accuracy)
    unweighted_accuracy = (recall_faulty + recall_healthy) / 2

    print(f"=== Metrics for {filename} ===")
    print(f"Total Healthy Samples: {total_negatives}")
    print(f"Total Faulty Samples:  {total_positives}")
    print("-" * 30)
    print(f"Recall (Healthy):      {recall_healthy:.2%} (True Negative Rate)")
    print(f"Recall (Faulty):       {recall_faulty:.2%} (True Positive Rate)")
    print(f"Unweighted Accuracy:   {unweighted_accuracy:.4f}")
    
    # 4. Normalize Confusion Matrix by Row (True Class)
    # This converts counts to percentages of the true label
    # axis=1 divides each row element by the row sum
    cm_normalized = joint_cm.astype('float') / joint_cm.sum(axis=1)[:, np.newaxis]

    # 5. Plot
    plt.figure(figsize=(8, 6))
    labels = ['Healthy (0)', 'Faulty (1)']
    
    # Create labels for the heatmap annotations
    # We format them as percentages "98.5%"
    annot_labels = np.array([
        [f"{cm_normalized[0,0]:.1%}", f"{cm_normalized[0,1]:.1%}"],
        [f"{cm_normalized[1,0]:.1%}", f"{cm_normalized[1,1]:.1%}"]
    ])

    sns.heatmap(cm_normalized, annot=annot_labels, fmt='', cmap='Blues', vmin=0, vmax=1,
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 16})
    
    plt.title('Joint Confusion Matrix (Row Normalized)', fontsize=16)
    plt.ylabel('True Class', fontsize=14)
    plt.xlabel('Predicted Class', fontsize=14)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    analyze_results('results/case1/cms.json')
    analyze_results('results/case2/cms.json')
    analyze_results('results/case3/cms.json')