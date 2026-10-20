# Known issues and modernization notes

This is an archived research implementation. The curated edition makes three targeted repairs, but does not attempt a full refactor.

1. **Dataset/permissions audit.** The transferred GitLab repository may include data and artifacts in the current tree or history. Review these for publication rights. The full CLI expects files in the documented `Datasets/` directory structure.
2. **AGIPD path issue.** The `AGIPD` option in `main.py` incorrectly selects `datasets.JAFFE()`, which is not defined; `datasets.py` also references a legacy folder name.
3. **KAZE mapping issue.** The `KAZE` detector and descriptor options in `main.py` currently instantiate the SIFT implementation instead of KAZE.
4. **SVM gamma argument (addressed).** The SVM wrapper now supplies `gamma` by keyword to `SVC`.
5. **BoVF bin count (addressed).** Histogram edges now span `0..K`, yielding one bin per visual word.
6. **Random Forest estimator count (addressed).** The wrapper now uses the supplied `n_estimators` parameter.
7. **No-keypoint fallback.** A legacy fallback inserts random descriptors when no keypoints are found; replace this with a documented deterministic strategy before reusing on another dataset.
8. **OpenCV extension methods.** The availability of `cv.xfeatures2d` functions depends on the OpenCV distribution; SURF may require a nonfree-enabled build.
9. **Dependency versions.** The exact original package versions are unknown; modern installations can require API compatibility updates.
10. **Output directory conventions.** The main script expects nested directories for generated files and serialized models; directories may need to be created locally if absent.

See [TESTING.md](TESTING.md) for the available software checks.
