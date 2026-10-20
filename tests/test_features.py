"""Unit tests for the existing feature extraction + BoVF implementation."""

import cv2 as cv
import numpy as np
import pytest

import features
import globals


def test_orb_extracts_real_descriptors_from_synthetic_image(monkeypatch):
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, (128, 128), dtype=np.uint8)
    orb = features.ORB()
    monkeypatch.setattr(globals, "detector", orb, raising=False)
    monkeypatch.setattr(globals, "descriptor", orb, raising=False)

    keypoints, descriptors = features.features(image)

    assert len(keypoints) > 0
    assert descriptors is not None
    assert descriptors.shape == (len(keypoints), 32)
    assert descriptors.dtype == np.uint8


def test_extract_features_indexes_images_with_descriptors():
    image_a = np.zeros((16, 16), dtype=np.uint8)
    image_b = np.ones((16, 16), dtype=np.uint8)

    def stub_feature_func(image):
        value = int(image[0, 0])
        return [object()], np.full((1, 4), value, dtype=np.uint8)

    result = features.extractFeatures([image_a, image_b], stub_feature_func)

    assert set(result) == {0, 1}
    np.testing.assert_array_equal(result[0], [[0, 0, 0, 0]])
    np.testing.assert_array_equal(result[1], [[1, 1, 1, 1]])


def test_group_all_features_flattens_nested_descriptor_sets():
    descriptor_map = {
        "Blank": {0: np.array([[1, 2], [3, 4]])},
        "Strong": {0: np.array([[5, 6]])},
    }
    flattened = np.asarray(features.groupAllFeatures(descriptor_map))
    np.testing.assert_array_equal(flattened, [[1, 2], [3, 4], [5, 6]])


@pytest.mark.parametrize("word_labels,k,expected", [
    ([0, 1, 1, 2, 3], 4, [1, 2, 1, 1]),
    ([3, 3, 3], 4, [0, 0, 0, 3]),
    ([0, 0, 0], 4, [3, 0, 0, 0]),
])
def test_bovf_histograms_count_every_visual_word(
    monkeypatch, word_labels, k, expected
):
    monkeypatch.setattr(globals, "K", k, raising=False)

    class Vocabulary:
        def predict(self, descriptors):
            assert descriptors.shape[0] == len(word_labels)
            return np.asarray(word_labels)

    descriptors = {0: np.zeros((len(word_labels), 2))}
    hist = features.generateHistograms(descriptors, Vocabulary())

    assert len(hist) == 1
    assert len(hist[0]) == k
    assert hist[0].sum() == len(word_labels)
    np.testing.assert_array_equal(hist[0], expected)


def test_bovf_empty_image_descriptor_set_yields_zero_histogram(monkeypatch):
    monkeypatch.setattr(globals, "K", 4, raising=False)

    class Vocabulary:
        def predict(self, descriptors):
            return np.empty((0,), dtype=int)

    hist = features.generateHistograms({0: np.empty((0, 32))}, Vocabulary())
    np.testing.assert_array_equal(hist[0], np.zeros(4, dtype=int))


def test_kmeans_learns_requested_number_of_visual_words(monkeypatch):
    monkeypatch.setattr(globals, "K", 3, raising=False)
    rng = np.random.default_rng(12)
    descriptors = np.vstack([
        rng.normal(loc=center, scale=0.1, size=(30, 4))
        for center in (-5.0, 0.0, 5.0)
    ])

    vocabulary = features.trainKMeans(descriptors)

    assert vocabulary.n_clusters == 3
    assert vocabulary.cluster_centers_.shape == (3, 4)
    assert len(vocabulary.predict(descriptors[:4])) == 4


def test_opencv_orb_exposes_expected_descriptor_size():
    assert features.ORB().descriptorSize() == 32
    assert isinstance(cv.__version__, str)
