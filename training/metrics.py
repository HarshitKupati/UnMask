"""
Evaluation metrics for the Deepfake Detection System.
Reference: Specification Document §12

Provides:
  - AUC-ROC (primary metric, §12.1)
  - F1 Score at threshold 0.5 (§12.1)
  - Average Precision (§12.1)
  - Per-manipulation-type breakdown (§12.2)
  - Optimal threshold calibration (§12.4)
  - Confusion matrix analysis (§12.5)
  - Temperature scaling (§12.4)
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
)


def compute_metrics(all_labels, all_logits, threshold=0.5):
    """
    Compute primary evaluation metrics (§12.1).

    Args:
        all_labels: numpy array of ground truth labels (0 or 1)
        all_logits: numpy array of raw logits
        threshold: decision threshold

    Returns:
        dict of metrics
    """
    all_labels = np.array(all_labels).flatten()
    # Binarize labels in case mixup produced continuous values during training
    all_labels = (all_labels >= 0.5).astype(int)
    all_logits = np.array(all_logits).flatten()

    # Convert logits to probabilities
    all_probs = 1.0 / (1.0 + np.exp(-all_logits))  # sigmoid

    # Binary predictions
    all_preds = (all_probs >= threshold).astype(int)

    metrics = {}

    # AUC-ROC
    try:
        metrics['auc'] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        metrics['auc'] = 0.0

    # F1 Score
    metrics['f1'] = f1_score(all_labels, all_preds, zero_division=0)

    # Average Precision
    try:
        metrics['ap'] = average_precision_score(all_labels, all_probs)
    except ValueError:
        metrics['ap'] = 0.0

    # Accuracy
    metrics['accuracy'] = accuracy_score(all_labels, all_preds)

    # Precision & Recall
    metrics['precision'] = precision_score(all_labels, all_preds, zero_division=0)
    metrics['recall'] = recall_score(all_labels, all_preds, zero_division=0)

    # Confusion Matrix (§12.5)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    metrics['confusion_matrix'] = cm
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)   # most dangerous (§12.5)
        metrics['true_positives'] = int(tp)
        # False Negative Rate
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return metrics


def compute_per_type_metrics(all_labels, all_logits, all_types, threshold=0.5):
    """
    Per-manipulation-type breakdown (§12.2).

    Args:
        all_labels: list/array of labels
        all_logits: list/array of logits
        all_types:  list of manipulation type strings
        threshold:  decision threshold

    Returns:
        dict mapping type_name → metrics_dict
    """
    all_labels = np.array(all_labels).flatten()
    all_logits = np.array(all_logits).flatten()

    unique_types = set(all_types)
    per_type = {}

    for mtype in unique_types:
        mask = np.array([t == mtype for t in all_types])
        if mask.sum() < 2:
            continue
        per_type[mtype] = compute_metrics(
            all_labels[mask], all_logits[mask], threshold
        )

    return per_type


def find_optimal_threshold(all_labels, all_logits):
    """
    Find optimal threshold by maximizing F1 (§12.4).
    Sweep from 0.1 to 0.9 in steps of 0.01.
    """
    all_labels = np.array(all_labels).flatten()
    all_logits = np.array(all_logits).flatten()
    all_probs = 1.0 / (1.0 + np.exp(-all_logits))

    best_threshold = 0.5
    best_f1 = 0.0

    for t in np.arange(0.1, 0.91, 0.01):
        preds = (all_probs >= t).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold, best_f1


def calibrate_temperature(all_labels, all_logits):
    """
    Temperature scaling calibration (§12.4).
    Find T that minimizes NLL on the validation set.
    """
    import torch
    import torch.nn as nn

    labels = torch.tensor(all_labels, dtype=torch.float32).view(-1, 1)
    logits = torch.tensor(all_logits, dtype=torch.float32).view(-1, 1)

    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    criterion = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    return temperature.item()


def get_confidence_label(probability):
    """
    Map sigmoid probability to confidence label (§13.3).

    Returns:
        (label, confidence_level) tuple
    """
    if probability >= 0.90:
        return 'FAKE', 'High confidence'
    elif probability >= 0.70:
        return 'FAKE', 'Medium confidence'
    elif probability >= 0.55:
        return 'FAKE', 'Low confidence (flag for human review)'
    elif probability >= 0.45:
        return 'UNCERTAIN', 'Uncertain (flag for human review)'
    elif probability >= 0.30:
        return 'REAL', 'Low confidence'
    else:
        return 'REAL', 'High confidence'
