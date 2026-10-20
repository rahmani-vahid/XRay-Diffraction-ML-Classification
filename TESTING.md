# Testing and continuous integration

This repository includes lightweight unit and integration tests for the research software. Tests construct **synthetic arrays and images in memory** and require no external diffraction-image dataset.

## Run locally

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Test areas

| Component | Test scenario |
| --- | --- |
| ORB feature extraction | Descriptor extraction and expected descriptor dimensionality |
| Feature bookkeeping | Indexing local descriptors and aggregating descriptor sets |
| Visual vocabulary | MiniBatch K-Means vocabulary creation |
| BoVF histogram representation | Bin boundaries, conserved descriptor counts, empty-descriptor handling |
| Classifier interfaces | Random Forest, SVM and Naive Bayes fit/predict smoke tests |
| Cross-module integration | Synthetic image → local descriptors → vocabulary → BoVF → classifier |
| File-output utility | Confusion-matrix figure writer saves a file to a temporary directory |
| Path handling and CLI | Dataset path definitions and command-line help |

The CI workflow in `.github/workflows/tests.yml` runs on pushes and pull requests using Python 3.11 and 3.12.

## Scope and limitations

- These are software checks using controlled synthetic inputs, not a substitute for scientific validation on experimental diffraction images.
- The suite does not cover every optional descriptor implementation, the complete CLI with external data, or every MLP training configuration.
- Some older OpenCV APIs and legacy code paths remain environment dependent; see [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## Small maintenance changes in this edition

1. `features.generateHistograms` now generates `K` bins with edges `0..K`.
2. `svm.train` passes the `gamma` parameter to `SVC` by keyword.
3. `rf.train` forwards the supplied `n_estimators` setting to `RandomForestClassifier`.

These updates improve reliability and compatibility without changing the overall research design.
