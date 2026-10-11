# MW-K-means implementation

This project implements the MW-K-means algorithm described in the paper
"K-means Clustering Algorithm and Its Improvement Research" and compares it
with classic K-means.

## Setup

```bash
pip install -r requirements.txt
```

## Python usage

```python
import numpy as np
from mw_kmeans import MWKMeans

X = np.array([
    [1.0, 2.0],
    [1.1, 2.1],
    [9.0, 9.5],
    [9.2, 9.1],
    [10.0, 9.8],
])

model = MWKMeans(n_clusters=2, beta=0.2, theta=0.8)
labels = model.fit_predict(X)
print(labels)
print(model.cluster_centers_)
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

MW-K-means uses 10 deterministic initializations by default and keeps the
solution with the lowest weighted within-cluster sum of squares. Change the
restart count or seed when needed:

```bash
python evaluate_mw_kmeans.py --dataset all --n-init 20 --random-state 7
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
