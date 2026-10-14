import numpy as np, time


def kmeans(X, k, c_init = None, max_iter=300, random_state=42):
    """
    K-means truyền thống theo nguyên lý trong bài báo.

    X: ma trận dữ liệu (N, d)
    k: số cụm
    max_iter: số vòng lặp tối đa
    random_state: seed khởi tạo
    """

    X = np.asarray(X, dtype=float)

    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X phải là ma trận dữ liệu không rỗng.")

    N, d = X.shape

    if not 1 <= k <= N:
        raise ValueError("k phải nằm trong khoảng từ 1 đến N.")

    if not np.all(np.isfinite(X)):
        raise ValueError("X không được chứa NaN hoặc infinity.")

    # Bước 1: Khởi tạo k tâm cụm từ dữ liệu
    rng = np.random.default_rng(random_state)
    indices = rng.choice(N, size=k, replace=False)
    centers = X[indices].copy() if c_init is None else c_init

    for iteration in range(max_iter):

        old_centers = centers.copy()

        # Bước 2: Tính bình phương khoảng cách Euclid
        # d(x_i, c_j) = sum_l (x_il - c_jl)^2
        distances = np.sum(
            (X[:, None, :] - centers[None, :, :]) ** 2,
            axis=2
        )

        # Bước 3: Gán mỗi điểm vào cụm có khoảng cách nhỏ nhất
        labels = np.argmin(distances, axis=1)

        # Bước 4: Cập nhật tâm bằng trung bình các điểm trong cụm
        for j in range(k):
            members = X[labels == j]

            if len(members) > 0:
                centers[j] = np.mean(members, axis=0)

        # Bước 5: Kiểm tra tâm cụm có còn thay đổi không
        if np.array_equal(centers, old_centers):
            break

    # Tính lại nhãn theo tâm cụm cuối cùng
    distances = np.sum(
        (X[:, None, :] - centers[None, :, :]) ** 2,
        axis=2
    )
    labels = np.argmin(distances, axis=1)

    # Hàm mục tiêu: tổng bình phương sai số trong cụm
    inertia = np.sum(
        (X - centers[labels]) ** 2
    )

    return {
        "labels": labels,
        "centers": centers,
        "inertia": float(inertia),
        "iterations": iteration + 1
    }
