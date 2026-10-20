"""Lightweight integration, output, and configuration checks."""

from pathlib import Path

import numpy as np

import datasets
import features
import globals
import save_figures
from timer import Timer


def test_dataset_paths_match_documented_layout():
    train, test = datasets.DiffraNet()
    assert Path(train).parts[-3:] == ("DiffraNet", "Dataset", "Train")
    assert Path(test).parts[-3:] == ("DiffraNet", "Dataset", "Test")


def test_timer_reports_elapsed_time():
    timer = Timer()
    assert str(timer) == "<not-measured>"
    with timer:
        sum(range(10_000))
    assert 0 <= float(timer) < 10


def test_synthetic_orb_kmeans_histogram_classification_pipeline(monkeypatch):
    """Image -> ORB -> KMeans -> BoVF -> RandomForest without real data."""
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(101)
    orb = features.ORB()
    monkeypatch.setattr(globals, "detector", orb, raising=False)
    monkeypatch.setattr(globals, "descriptor", orb, raising=False)
    monkeypatch.setattr(globals, "K", 6, raising=False)

    # Distinct noise intensities are used only to exercise the flow, not
    # to claim scientific predictive performance.
    images = [rng.integers(0, 256, (96, 96), dtype=np.uint8) for _ in range(10)]
    descriptor_sets = features.extractFeatures(images, features.features)
    assert len(descriptor_sets) == len(images)
    vocabulary = features.trainKMeans(np.asarray(features.groupAllFeatures({"synthetic": descriptor_sets})))
    histograms = np.asarray(features.generateHistograms(descriptor_sets, vocabulary))

    assert histograms.shape == (len(images), 6)
    assert np.all(histograms >= 0)
    for i in range(len(images)):
        assert histograms[i].sum() == len(descriptor_sets[i])

    y = np.array(["A", "B"] * 5)
    model = RandomForestClassifier(n_estimators=5, random_state=11).fit(histograms, y)
    predictions = model.predict(histograms[:2])
    assert predictions.shape == (2,)
    assert set(predictions).issubset({"A", "B"})


def test_confusion_matrix_figure_saves_to_disk(monkeypatch, tmp_path):
    folder = tmp_path / "Figures" / "Datasets" / "DiffraNet"
    folder.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    labels = ["Blank", "No-Crystal", "Weak", "Good", "Strong"]
    monkeypatch.setattr(globals, "test_feature_vec", [np.zeros((5, 2)), labels], raising=False)
    monkeypatch.setattr(globals, "num_classes", 5, raising=False)

    save_figures.confusionMatrix(labels, "ORB", "DiffraNet", "RF")

    path = folder / "ORB-RF-confusion-matrix.png"
    assert path.is_file()
    assert path.stat().st_size > 1_000


def test_cli_help_runs_without_datasets():
    """The documented command-line entry point can be inspected offline."""
    import subprocess
    import sys

    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--MLmodel" in result.stdout
    assert "--detector" in result.stdout
    assert "--dataset" in result.stdout
