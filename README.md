# Clustering Algorithm comparation

This project implements the different clustering algorithm and compares it
with each other.

## Setup

```bash
pip install -r requirements.txt
```

## Evaluate datasets

```bash
python evaluate_mw_kmeans.py
python evaluate_classic_kmeans.py
```

MW-K-means writes to `results/results.csv` and `results/results.png`.
Classic K-means writes to `results/classic_kmeans_results.csv` and
`results/classic_kmeans_results.png`. Both scripts default to all four
datasets: Iris, Wine, Breast Cancer, and Yeast. Yeast is loaded from the
classic OpenML dataset ID 181 (1,484 samples, 8 numeric features, 10 classes);
the first Yeast run requires internet access and scikit-learn caches the
download. Select one dataset or all:

```bash
python evaluate_mw_kmeans.py --dataset iris
python evaluate_classic_kmeans.py --dataset wine
python evaluate_mw_kmeans.py --dataset yeast
python evaluate_mw_kmeans.py --dataset all
```


Output paths can be overridden with `--output` and `--plot`; missing output
directories are created automatically.

The benchmark scales features to `[0, 1]`, uses the ground-truth cluster count,
and assigns every sample to a cluster. It reports class-aligned Accuracy,
macro Precision/Recall/F1, Adjusted Rand Index (ARI), Normalized Mutual
Information (NMI), runtime, and iteration count. Class alignment uses an
optimal one-to-one mapping because cluster IDs do not inherently match the
ground-truth label IDs.

Both MW-K-means and classic K-means use the same features and `[0, 1]` scaling
for each dataset, so their benchmark comparison does not give one model a
hand-selected feature subset.

## Notes

- `beta` controls the density threshold: `MinPts = ceil(beta * N)`
- `theta` scales neighborhood radius `epsilon` based on average pairwise distance
- `MWKMeans` treats low-density points as noise by default; the benchmark sets
  `remove_noise=False` so every point is included in the evaluation
- MW-K-means computes the mean pairwise distance in triangular chunks and
  updates nearest-center distances incrementally; density estimation remains
  quadratic in the number of samples
- Multiple initializations choose using clustering inertia only, without
  consulting dataset labels; better inertia does not guarantee higher ARI or
  accuracy on every dataset
- Classic K-means uses scikit-learn's `k-means++` initialization, 10 restarts,
  and a fixed random state of 42 by default

Yeast source: [OpenML dataset 181](https://www.openml.org/d/181).
