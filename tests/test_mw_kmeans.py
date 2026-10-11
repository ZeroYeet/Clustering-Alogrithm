from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import benchmark_data
import mw_kmeans
from benchmark_data import expand_dataset_selection
from evaluate_classic_kmeans import evaluate_dataset as evaluate_classic_dataset
from evaluate_mw_kmeans import RESULTS_DIR, evaluate_dataset, save_csv, save_plot
from mw_kmeans import MWKMeans


def test_mw_kmeans_clusters_two_blobs():
    rng = np.random.default_rng(7)
    cluster_a = rng.normal(loc=[0.0, 0.0], scale=0.4, size=(60, 2))
    cluster_b = rng.normal(loc=[5.0, 5.0], scale=0.4, size=(60, 2))
    X = np.vstack([cluster_a, cluster_b])

    model = MWKMeans(n_clusters=2, beta=0.05, theta=0.6, max_iter=200, tol=1e-6)
    labels = model.fit_predict(X)

    unique_labels = set(labels.tolist())
    assert unique_labels.issubset({-1, 0, 1})
    assert len(set(labels[labels != -1])) == 2
    assert model.cluster_centers_.shape == (2, 2)
    assert model.weights_.shape == (2,)


def test_pairwise_distances_matches_euclidean_distance():
    X = np.array([[0.0, 0.0], [3.0, 4.0], [-1.0, 2.0]])
    distances = MWKMeans(n_clusters=2)._pairwise_distances(X)

    np.testing.assert_allclose(
        distances,
        np.array([[0.0, 5.0, np.sqrt(5)], [5.0, 0.0, np.sqrt(20)], [np.sqrt(5), np.sqrt(20), 0.0]]),
    )


def test_mean_pairwise_distance_matches_upper_triangle_mean_across_chunks(monkeypatch):
    monkeypatch.setattr(mw_kmeans, "_PAIRWISE_CHUNK_SIZE", 2)
    X = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [-1.0, 2.0],
            [1.0, 1.0],
            [2.0, -1.0],
        ]
    )
    model = MWKMeans(n_clusters=2)
    expected = model._pairwise_distances(X)[np.triu_indices(len(X), k=1)].mean()

    assert model._mean_pairwise_distance(X) == pytest.approx(expected)


def test_initial_centers_keep_greedy_max_min_selection_with_duplicate_rows():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [5.0, 0.0],
            [5.0, 0.0],
            [0.0, 2.0],
        ]
    )
    density = np.array([10, 0, 0, 0, 0])
    centers = MWKMeans(n_clusters=3)._select_initial_centers(X, density)

    np.testing.assert_array_equal(
        centers,
        np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 2.0]]),
    )


def test_weighted_distance_matrix_matches_per_center_distances():
    X = np.array([[0.0, 0.0], [3.0, 4.0], [-1.0, 2.0]])
    centers = np.array([[1.0, 1.0], [2.0, 3.0]])
    weights = np.array([0.25, 0.75])
    model = MWKMeans(n_clusters=2)
    expected = np.column_stack(
        [
            model._weighted_distance(X, center, weights) ** 2
            for center in centers
        ]
    )

    np.testing.assert_allclose(
        model._weighted_distance_matrix(X, centers, weights),
        expected,
    )


def test_fit_computes_density_parameters_once(monkeypatch):
    model = MWKMeans(n_clusters=2, remove_noise=False)
    original_infer = model._infer_density_params
    calls = 0

    def counted_infer(X):
        nonlocal calls
        calls += 1
        return original_infer(X)

    monkeypatch.setattr(model, "_infer_density_params", counted_infer)
    model.fit(np.array([[0.0], [0.1], [5.0], [5.1]]))

    assert calls == 1


def test_multiple_initializations_are_reproducible_and_not_worse_than_first():
    rng = np.random.default_rng(21)
    X = np.vstack(
        [
            rng.normal(loc=[0.0, 0.0], scale=0.8, size=(45, 2)),
            rng.normal(loc=[3.0, 0.0], scale=0.8, size=(45, 2)),
            rng.normal(loc=[1.5, 2.5], scale=0.8, size=(45, 2)),
        ]
    )
    single_start = MWKMeans(n_clusters=3, remove_noise=False, n_init=1)
    multi_start = MWKMeans(
        n_clusters=3,
        remove_noise=False,
        n_init=8,
        random_state=12,
    )

    single_start.fit(X)
    multi_start.fit(X)
    repeated_fit = MWKMeans(
        n_clusters=3,
        remove_noise=False,
        n_init=8,
        random_state=12,
    ).fit(X)

    assert multi_start.inertia_ <= single_start.inertia_ + 1e-12
    assert multi_start.inertia_ == pytest.approx(repeated_fit.inertia_)
    np.testing.assert_array_equal(multi_start.labels_, repeated_fit.labels_)
    np.testing.assert_array_equal(multi_start.labels_, multi_start.predict(X))


def test_reported_inertia_matches_final_assignments():
    rng = np.random.default_rng(42)
    X = np.vstack(
        [
            rng.normal(loc=[0.0, 0.0], scale=0.7, size=(40, 2)),
            rng.normal(loc=[3.0, 0.0], scale=0.7, size=(40, 2)),
            rng.normal(loc=[1.5, 2.5], scale=0.7, size=(40, 2)),
        ]
    )
    model = MWKMeans(n_clusters=3, remove_noise=False, n_init=12).fit(X)
    weighted_data = X * np.sqrt(model.weights_)
    weighted_centers = model.cluster_centers_ * np.sqrt(model.weights_)
    distances = model._squared_distance_matrix(
        weighted_data,
        weighted_centers,
    )
    selected_distances = distances[np.arange(len(X)), model.labels_]

    assert model.inertia_ == pytest.approx(selected_distances.sum())


def test_mw_kmeans_rejects_non_positive_n_init():
    with pytest.raises(ValueError, match="n_init must be positive"):
        MWKMeans(n_clusters=2, n_init=0)


def test_mw_kmeans_predict_uses_trained_centers():
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [5.0, 5.0],
        [5.0, 5.1],
    ])

    model = MWKMeans(n_clusters=2, beta=0.35, theta=0.8)
    model.fit(X)
    preds = model.predict(np.array([[0.05, 0.05], [4.9, 5.1]]))

    assert preds.shape == (2,)
    assert set(preds.tolist()).issubset({0, 1})


def test_evaluate_iris_and_wine_returns_complete_dataset_metrics():
    expected_data = {
        "iris": (150, 3, 4),
        "wine": (178, 3, 13),
        "cancer": (569, 2, 30),
    }

    for dataset_name, (sample_count, cluster_count, feature_count) in expected_data.items():
        result = evaluate_dataset(dataset_name)

        assert result["samples"] == sample_count
        assert result["clusters"] == cluster_count
        assert result["features"] == feature_count
        assert result["noise_points"] == 0
        for metric in (
            "accuracy_percent",
            "precision_macro_percent",
            "recall_macro_percent",
            "f1_macro_percent",
        ):
            assert 0 <= result[metric] <= 100
        assert -1 <= result["ari"] <= 1
        assert 0 <= result["nmi"] <= 1
        assert result["runtime_seconds"] >= 0
        assert result["iterations"] > 0


def test_both_iris_evaluators_use_all_features():
    mw_result = evaluate_dataset("iris")
    classic_result = evaluate_classic_dataset("iris")

    assert mw_result["features"] == classic_result["features"] == 4


def test_evaluate_dataset_rejects_unknown_dataset():
    try:
        evaluate_dataset("unknown")
    except ValueError as error:
        assert "iris, wine, cancer" in str(error)
    else:
        raise AssertionError("Expected ValueError for an unknown dataset")


def test_dataset_selection_includes_supported_datasets_only():
    assert expand_dataset_selection("all") == ("iris", "wine", "cancer", "yeast")
    assert expand_dataset_selection("iris") == ("iris",)
    for removed_dataset in ("s1", "a1", "s-sets", "a-sets", "uef"):
        with pytest.raises(ValueError):
            expand_dataset_selection(removed_dataset)


def test_load_openml_yeast_dataset_by_fixed_id(monkeypatch):
    raw_features = np.arange(1484 * 8, dtype=float).reshape(1484, 8)
    raw_features %= 101
    labels = np.asarray([f"class-{index % 10}" for index in range(1484)])
    calls = []

    def fake_fetch_openml(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=raw_features, target=labels)

    monkeypatch.setattr(benchmark_data, "fetch_openml", fake_fetch_openml)
    features, loaded_labels = benchmark_data.load_benchmark_data("yeast")

    assert calls == [{"data_id": 181, "as_frame": False, "parser": "auto"}]
    assert features.shape == (1484, 8)
    assert np.isfinite(features).all()
    np.testing.assert_allclose(features.min(axis=0), 0)
    np.testing.assert_allclose(features.max(axis=0), 1)
    np.testing.assert_array_equal(loaded_labels, labels)


def test_both_evaluators_run_openml_yeast_without_network(monkeypatch):
    rng = np.random.default_rng(17)
    features = rng.random((1484, 8))
    labels = np.asarray([f"class-{index % 10}" for index in range(1484)])

    def fake_fetch_openml(data_id, as_frame, parser):
        assert (data_id, as_frame, parser) == (181, False, "auto")
        return SimpleNamespace(data=features, target=labels)

    monkeypatch.setattr(
        benchmark_data,
        "fetch_openml",
        fake_fetch_openml,
    )

    mw_result = evaluate_dataset("yeast")
    classic_result = evaluate_classic_dataset("yeast", n_init=2)

    for result in (mw_result, classic_result):
        assert result["samples"] == 1484
        assert result["features"] == 8
        assert result["clusters"] == 10
        assert result["noise_points"] == 0


def test_classic_kmeans_evaluates_all_shared_datasets():
    expected_data = {"iris": (150, 3), "wine": (178, 3), "cancer": (569, 2)}

    for dataset_name, (sample_count, cluster_count) in expected_data.items():
        result = evaluate_classic_dataset(dataset_name)

        assert result["samples"] == sample_count
        assert result["clusters"] == cluster_count
        assert result["noise_points"] == 0
        assert 0 <= result["accuracy_percent"] <= 100
        assert -1 <= result["ari"] <= 1
        assert 0 <= result["nmi"] <= 1
        assert result["runtime_seconds"] >= 0
        assert result["iterations"] > 0


def test_classic_kmeans_rejects_invalid_n_init():
    try:
        evaluate_classic_dataset("iris", n_init=0)
    except ValueError as error:
        assert "n_init must be positive" in str(error)
    else:
        raise AssertionError("Expected ValueError for n_init=0")


def test_save_csv_exports_evaluation_metrics(tmp_path):
    result = evaluate_dataset("iris")
    output_path = tmp_path / "results" / "evaluation.csv"

    save_csv([result], output_path)

    exported = output_path.read_text(encoding="utf-8").splitlines()
    assert exported[0].startswith("dataset,samples,features,clusters,")
    assert exported[1].startswith("iris,150,4,3,")


def test_save_plot_exports_chart_image(tmp_path):
    results = [evaluate_dataset("iris"), evaluate_dataset("cancer")]
    output_path = tmp_path / "results" / "metrics.png"

    save_plot(results, output_path)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_default_output_paths_point_to_results_directory():
    assert RESULTS_DIR.name == "results"
    assert RESULTS_DIR.parent == Path(__file__).resolve().parents[1]


def test_save_plot_rejects_empty_results(tmp_path):
    try:
        save_plot([], tmp_path / "empty.png")
    except ValueError as error:
        assert "At least one dataset result" in str(error)
    else:
        raise AssertionError("Expected ValueError when plotting no results")
