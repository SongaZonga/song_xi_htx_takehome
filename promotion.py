def should_promote(metrics, previous_metrics=None, min_accuracy=0.88, min_auc=0.95):
    if metrics["eval_accuracy"] < min_accuracy:
        return False, f"Accuracy {metrics['eval_accuracy']:.3f} below floor {min_accuracy}"
    if metrics["eval_auc"] < min_auc:
        return False, f"AUC {metrics['eval_auc']:.3f} below floor {min_auc}"
    if previous_metrics is not None:
        if metrics["eval_accuracy"] < previous_metrics["eval_accuracy"]:
            return False, "Accuracy regressed vs baseline"
        if metrics["eval_auc"] < previous_metrics["eval_auc"]:
            return False, "AUC regressed vs baseline"
    return True, "Model promoted"