"""Classifier tests on synthetic labeled feature vectors, not DiffraNet."""

import numpy as np
import pytest

import nb
import rf
import svm


@pytest.mark.parametrize("model_name", ["RF", "NB", "SVM"])
def test_train_and_predict_model(supervised_data, model_name):
    _, _, X_test, y_test, output = supervised_data

    if model_name == "RF":
        model = rf.train(n_estimators=15, descriptor_name="ORB", model_name="RF")
        predicted = rf.test(model, descriptor_name="ORB", model_name="RF")
        assert model.n_estimators == 15
    elif model_name == "NB":
        model = nb.train(descriptor_name="ORB", model_name="NB")
        predicted = nb.test(model, descriptor_name="ORB", model_name="NB")
    else:
        model = svm.train(gama=0.1, descriptor_name="ORB", model_name="SVM")
        predicted = svm.test(model, descriptor_name="ORB", model_name="SVM")
        assert model.gamma == 0.1

    assert predicted.shape == y_test.shape
    assert set(predicted).issubset({"Blank", "Strong"})
    assert np.mean(predicted == y_test) >= 0.9  # easy synthetic separable data
    assert "train score" in output.getvalue().lower()
    assert "test score" in output.getvalue().lower()


def test_random_forest_is_reproducible_with_fixed_seed_on_synthetic_data(
    supervised_data,
):
    from sklearn.ensemble import RandomForestClassifier
    X_train, y_train, X_test, _, _ = supervised_data
    a = RandomForestClassifier(n_estimators=12, random_state=7).fit(X_train, y_train)
    b = RandomForestClassifier(n_estimators=12, random_state=7).fit(X_train, y_train)
    np.testing.assert_array_equal(a.predict(X_test), b.predict(X_test))
