from sklearn.metrics import accuracy_score, f1_score, average_precision_score
import numpy as np

def calculate_metrics(y_true, y_probs, threshold=0.5):
    """
    Calculate evaluation metrics for binary classification using a specified threshold.

    Args:
        y_true (np.array): True binary labels.
        y_probs (np.array): Predicted probabilities.
        threshold (float): Threshold to convert probabilities to binary predictions.

    Returns:
        dict: Dictionary containing accuracy, F1 score, average precision, real accuracy, 
              fake accuracy, and the threshold used.
    """
    # Convert probabilities to binary predictions using the specified threshold
    y_pred = (y_probs > threshold).astype(int)

    # Calculate metrics
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "real_accuracy": round(accuracy_score(y_true[y_true == 0], y_pred[y_true == 0]), 4) if (y_true == 0).any() else 0.0,
        "fake_accuracy": round(accuracy_score(y_true[y_true == 1], y_pred[y_true == 1]), 4) if (y_true == 1).any() else 0.0,
        "f1": round(f1_score(y_true, y_pred), 4),
        "average_precision": round(average_precision_score(y_true, y_probs), 4),
        "threshold": round(threshold, 4)
    }

    # Convert all values to native Python types
    return {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in metrics.items()}


def find_best_threshold(y_true, y_probs):
    """
    Find the best threshold for binary classification to maximize accuracy.

    Args:
        y_true (np.array): True binary labels.
        y_probs (np.array): Predicted probabilities.

    Returns:
        float: The best threshold that maximizes accuracy.
    """
    thresholds = np.unique(y_probs)
    best_accuracy = 0.0
    best_threshold = 0.5

    for thresh in thresholds:
        y_pred = (y_probs > thresh).astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = thresh

    return round(float(best_threshold), 4)
