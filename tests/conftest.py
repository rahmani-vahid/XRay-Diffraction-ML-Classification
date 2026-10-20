"""Test fixtures: deterministic, synthetic inputs only (no experimental data)."""

import io

import numpy as np
import pytest

import globals


@pytest.fixture
def supervised_data(monkeypatch):
    """Linearly separated, reproducible examples for classifier smoke tests."""
    rng = np.random.default_rng(2021)
    X_train = np.vstack((
        rng.normal(loc=-2.0, scale=0.3, size=(30, 8)),
        rng.normal(loc=2.0, scale=0.3, size=(30, 8)),
    ))
    y_train = np.array(["Blank"] * 30 + ["Strong"] * 30)
    X_test = np.vstack((
        rng.normal(loc=-2.0, scale=0.3, size=(10, 8)),
        rng.normal(loc=2.0, scale=0.3, size=(10, 8)),
    ))
    y_test = np.array(["Blank"] * 10 + ["Strong"] * 10)
    output = io.StringIO()
    monkeypatch.setattr(globals, "train_feature_vec", [X_train, y_train], raising=False)
    monkeypatch.setattr(globals, "test_feature_vec", [X_test, y_test], raising=False)
    monkeypatch.setattr(globals, "file", output, raising=False)
    return X_train, y_train, X_test, y_test, output
