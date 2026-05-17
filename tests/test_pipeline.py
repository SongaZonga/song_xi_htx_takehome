import pytest
from promotion import should_promote

def test_should_promote_rejects_low_accuracy():
    metrics = {"eval_accuracy": 0.75, "eval_auc": 0.97}
    promoted, reason = should_promote(metrics)
    assert promoted == False
    assert "Accuracy" in reason

def test_should_promote_rejects_regression():
    previous = {"eval_accuracy": 0.92, "eval_auc": 0.97}
    current  = {"eval_accuracy": 0.90, "eval_auc": 0.98}  # accuracy dropped
    promoted, reason = should_promote(current, previous_metrics=previous)
    assert promoted == False
    assert "regressed" in reason

def test_should_promote_passes_when_all_conditions_met():
    metrics = {"eval_accuracy": 0.92, "eval_auc": 0.97}
    promoted, reason = should_promote(metrics)
    assert promoted == True