# Clustering Algorithm comparation

This project implements different clustering algorithms and compares them
with each other.

## Setup

```bash
pip install -r requirements.txt
```

## Evaluate datasets

```bash
python -m model.evaluate_mw_kmeans
python -m model.evaluate_classic_kmeans
python -m model.evaluate_dca
python evaluate_all_models.py
```

The individual model scripts default
to all four datasets: Iris, Wine, Breast Cancer, and Yeast. Yeast is loaded
from the classic OpenML dataset ID 181 (1,484 samples, 8 numeric features,
10 classes); the first Yeast run requires internet access and scikit-learn
caches the download. `evaluate_all_models.py` runs all three models and writes
one row for each model and dataset to `results/all_models_results.csv` by
default. Select one dataset or all; the program also exports a comparison chart to
`results/all_models_results.png`. The combined output includes Accuracy,
Precision, Recall, F1, runtime, and iteration count (ARI/NMI are omitted).

```bash
python -m model.evaluate_mw_kmeans --dataset iris
python -m model.evaluate_classic_kmeans --dataset wine
python -m model.evaluate_mw_kmeans --dataset yeast
python -m model.evaluate_mw_kmeans --dataset all
python evaluate_all_models.py --dataset iris
```

To evaluate a tabular CSV hosted at a direct HTTP(S) download URL, provide
`--dataset-url` instead of `--dataset`. The target defaults to the last column
and can be selected by name or zero-based index with `--target-column`.
Numeric features are median-imputed, categorical features are mode-imputed and
one-hot encoded, and the resulting features are scaled to `[0, 1]`. Missing
target values are rejected. Columns whose names clearly identify them as IDs
(for example `User ID`) are excluded automatically. A header is required when
the feature columns are categorical; CSV, TSV, semicolon, and pipe delimiters
are detected automatically. Use `--no-header` for headerless files with
categorical features when automatic header detection is ambiguous. For example:

```bash
python -m model.evaluate_classic_kmeans --dataset-url "https://example.com/data.csv" --target-column "column_name"
python -m model.evaluate_mw_kmeans --dataset-url "https://example.com/data.csv" --target-column "column_name"
python -m model.evaluate_dca --dataset-url "https://example.com/data.csv" --target-column "column_name"
python evaluate_all_models.py --dataset-url "https://example.com/data.csv" --target-column "column_name"
```

This supports direct links to tabular files, not dataset landing pages (such as
a Kaggle dataset page). Numeric and categorical columns are supported; formats
such as Excel, archives, and images are not CSV tabular inputs.


Output paths can be overridden with `--output` and `--plot`; missing output
directories are created automatically.


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
- Classic K-means uses the project's implementation in `kmeans.py` and runs
  10 random initializations by default, selecting the result with lowest inertia

Yeast source: [OpenML dataset 181](https://www.openml.org/d/181).
