from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits

from benchmark_data import (
    DATASET_SELECTIONS,
    expand_dataset_selection,
    load_benchmark_data,
)
from model.dca import ip_dca
from model.evaluate_mw_kmeans import RESULTS_DIR, METRIC_FIELDS, score_clustering
from model.kmeans import kmeans
from model.mw_kmeans import MWKMeans


ALL_MODEL_FIELDS = (
    "model",
    *(field for field in METRIC_FIELDS if field not in {"ari", "nmi"}),
)


def evaluate_dataset(
    dataset_name: str,
    *,
    dataset_url: str | None = None,
    target_column: str | None = None,
    has_header: bool | None = None,
    beta: float = 0.05,
    theta: float = 0.5,
    n_init: int = 10,
    max_iter: int = 300,
    dca_t: float = 10.0,
    dca_max_iter: int = 1000,
    tol: float = 1e-6,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Evaluate all three project clustering models on one loaded dataset."""
    features, true_labels = load_benchmark_data(
        dataset_name,
        url=dataset_url,
        target_column=target_column,
        has_header=has_header,
    )
    n_clusters = len(np.unique(true_labels))
    results: list[dict[str, Any]] = []

    mw_model = MWKMeans(
        n_clusters=n_clusters,
        beta=beta,
        theta=theta,
        max_iter=max_iter,
        remove_noise=False,
        n_init=n_init,
        random_state=random_state,
    )
    mw_model.fit(features)
    start = time.perf_counter()
    mw_labels = mw_model.fit_predict(features)
    mw_runtime = time.perf_counter() - start
    results.append(
        _score_model(
            "MW-K-means",
            dataset_name,
            true_labels,
            mw_labels,
            features,
            n_clusters,
            mw_runtime,
            mw_model.n_iter_,
        )
    )

    with threadpool_limits(limits=1):
        start = time.perf_counter()
        best_kmeans = None
        for init_index in range(n_init):
            candidate = kmeans(
                features,
                k=n_clusters,
                max_iter=max_iter,
                random_state=random_state + init_index,
            )
            if best_kmeans is None or candidate["inertia"] < best_kmeans["inertia"]:
                best_kmeans = candidate
        kmeans_runtime = time.perf_counter() - start

    if best_kmeans is None:
        raise RuntimeError("K-means did not produce a result")
    results.append(
        _score_model(
            "K-means",
            dataset_name,
            true_labels,
            best_kmeans["labels"],
            features,
            n_clusters,
            kmeans_runtime,
            best_kmeans["iterations"],
        )
    )

    start = time.perf_counter()
    dca_result = ip_dca(
        features,
        n_clusters=n_clusters,
        t=dca_t,
        max_iter=dca_max_iter,
        tol=tol,
        random_state=random_state,
    )
    dca_runtime = time.perf_counter() - start
    results.append(
        _score_model(
            "DCA",
            dataset_name,
            true_labels,
            dca_result["labels"],
            features,
            n_clusters,
            dca_runtime,
            dca_result["iterations"],
        )
    )
    return results


def _score_model(
    model_name: str,
    dataset_name: str,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    features: np.ndarray,
    n_clusters: int,
    runtime_seconds: float,
    iterations: int,
) -> dict[str, Any]:
    result = score_clustering(
        dataset_name,
        true_labels,
        predicted_labels,
        n_features=features.shape[1],
        n_clusters=n_clusters,
        runtime_seconds=runtime_seconds,
        iterations=iterations,
    )
    return {"model": model_name, **result}


def save_combined_csv(
    results: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Write all model and dataset results to one CSV file."""
    output_file_path = Path(output_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with output_file_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=ALL_MODEL_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: result[field] for field in ALL_MODEL_FIELDS}
            for result in results
        )


def save_combined_plot(
    results: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Save model metric and runtime comparisons as a PNG chart."""
    if not results:
        raise ValueError("At least one model result is required to create a plot")

    names = [f"{row['dataset']}\n{row['model']}" for row in results]
    metric_fields = (
        "accuracy_percent",
        "precision_macro_percent",
        "recall_macro_percent",
        "f1_macro_percent",
    )
    metric_names = ("Accuracy", "Precision", "Recall", "F1")
    positions = np.arange(len(results))
    width = 0.18

    figure, (metrics_axis, runtime_axis) = plt.subplots(
        2,
        1,
        figsize=(max(10, 1.4 * len(results)), 8),
        gridspec_kw={"height_ratios": [2, 1]},
    )
    figure.suptitle("Clustering model comparison", fontsize=15)
    for index, (field, label) in enumerate(zip(metric_fields, metric_names)):
        values = [row[field] for row in results]
        bars = metrics_axis.bar(
            positions + (index - 1.5) * width,
            values,
            width,
            label=label,
        )
        metrics_axis.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)

    metrics_axis.set_ylabel("Score (%)")
    metrics_axis.set_ylim(0, 115)
    metrics_axis.set_xticks(positions, names, rotation=30, ha="right")
    metrics_axis.legend(ncols=4, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    metrics_axis.grid(axis="y", linestyle=":", alpha=0.5)

    runtimes = [row["runtime_seconds"] for row in results]
    runtime_bars = runtime_axis.bar(positions, runtimes, color="slateblue")
    runtime_axis.bar_label(runtime_bars, fmt="%.4f", fontsize=7, padding=2)
    runtime_axis.set_ylabel("Seconds")
    runtime_axis.set_title("Execution time")
    runtime_axis.set_xticks(positions, names, rotation=30, ha="right")
    runtime_axis.set_ylim(0, max(runtimes, default=0.0) * 1.2 or 1.0)
    runtime_axis.grid(axis="y", linestyle=":", alpha=0.5)

    figure.tight_layout(rect=(0, 0, 1, 0.96))
    plot_path = Path(output_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare MW-K-means, K-means, and DCA in one CSV file."
    )
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--dataset",
        choices=DATASET_SELECTIONS,
        default="all",
        help="Built-in dataset to evaluate, or all datasets (default: all).",
    )
    dataset_group.add_argument(
        "--dataset-url",
        help="Direct HTTP(S) URL to a tabular CSV dataset.",
    )
    parser.add_argument(
        "--target-column",
        help="Target column name or zero-based index (default: last column).",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Treat the CSV as headerless.",
    )
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--dca-t", type=float, default=10.0)
    parser.add_argument("--dca-max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "all_models_results.csv",
        help="Combined CSV output path.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=RESULTS_DIR / "all_models_results.png",
        help="Combined comparison chart output path.",
    )
    args = parser.parse_args()

    if args.n_init <= 0:
        parser.error("--n-init must be positive")
    if not args.dataset_url and (args.target_column is not None or args.no_header):
        parser.error("--target-column and --no-header require --dataset-url")

    if args.dataset_url:
        dataset_names = ("custom",)
    else:
        dataset_names = expand_dataset_selection(args.dataset)
    results = [
        row
        for dataset_name in dataset_names
        for row in evaluate_dataset(
            dataset_name,
            dataset_url=args.dataset_url,
            target_column=args.target_column,
            has_header=False if args.no_header else None,
            beta=args.beta,
            theta=args.theta,
            n_init=args.n_init,
            max_iter=args.max_iter,
            dca_t=args.dca_t,
            dca_max_iter=args.dca_max_iter,
            tol=args.tol,
            random_state=args.random_state,
        )
    ]
    save_combined_csv(results, args.output)
    save_combined_plot(results, args.plot)
    print(f"Saved {len(results)} model results to: {args.output}")
    print(f"Saved comparison chart to: {args.plot}")
    print("Dataset | Model | Accuracy % | Precision % | Recall % | F1 % | Time (s) | Iterations")
    for result in results:
        print(
            f"{result['dataset']} | {result['model']} | "
            f"{result['accuracy_percent']:.2f} | "
            f"{result['precision_macro_percent']:.2f} | "
            f"{result['recall_macro_percent']:.2f} | "
            f"{result['f1_macro_percent']:.2f} | "
            f"{result['runtime_seconds']:.4f} | "
            f"{result['iterations']}"
        )


if __name__ == "__main__":
    main()
