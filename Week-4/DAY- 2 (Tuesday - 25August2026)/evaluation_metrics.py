import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


class EvaluationMetrics:
    """
    Computes and formats classification evaluation metrics:
    - Confusion Matrix
    - Accuracy, Precision, Recall, F1 (macro/weighted)
    - ROC-AUC (binary & multi-class with One-vs-Rest)
    """

    @staticmethod
    def evaluate(y_true, y_pred, y_proba=None):
        """
        Calculates a full suite of classification metrics.
        """
        classes = np.unique(y_true)
        is_binary = len(classes) == 2

        acc = accuracy_score(y_true, y_pred)
        avg_mode = "binary" if is_binary else "weighted"

        precision = precision_score(y_true, y_pred, average=avg_mode, zero_division=0)
        recall = recall_score(y_true, y_pred, average=avg_mode, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=avg_mode, zero_division=0)

        # ROC-AUC calculation
        auc = None
        if y_proba is not None:
            try:
                if is_binary:
                    # positive class probability
                    pos_proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                    auc = roc_auc_score(y_true, pos_proba)
                else:
                    auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
            except Exception:
                auc = None

        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, zero_division=0)

        return {
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": auc,
            "Confusion Matrix": cm,
            "Classification Report": report
        }
