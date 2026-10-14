# import numpy as np

# # ============================================================
# # 1. Projection onto the probability simplex
# #    Formula: Proj_Delta(y)
# #    Delta = {u : u_i >= 0, sum(u_i) = 1}
# # ============================================================

# def project_simplex(y):
#     y = np.asarray(y, dtype=float)

#     sorted_y = np.sort(y)[::-1]
#     cumulative = np.cumsum(sorted_y) - 1.0
#     indices = np.arange(1, len(y) + 1)

#     valid = sorted_y - cumulative / indices > 0
#     rho_idx = np.nonzero(valid)[0][-1]
#     theta = cumulative[rho_idx] / (rho_idx + 1)

#     return np.maximum(y - theta, 0.0)


# # ============================================================
# # 2. Projection onto the Euclidean ball
# #    Formula: Proj_R(z)
# # ============================================================

# def project_ball(z, radius):
#     norm_z = np.linalg.norm(z)

#     if norm_z <= radius:
#         return z.copy()

#     if norm_z == 0:
#         return z.copy()

#     return (radius / norm_z) * z


# # ============================================================
# # 3. MSSC objective and exact-penalty objective
# #
# # F(U,V) = sum_i sum_k u[i,k]^2 * ||x[k] - v[i]||^2
# #
# # p(U) = sum_i sum_k u[i,k] * (1 - u[i,k])
# #
# # F1(U,V) = F(U,V) + t * p(U)
# # ============================================================

# def mssc_objective(X, U, V):
#     diff = X[None, :, :] - V[:, None, :]
#     distances_sq = np.sum(diff ** 2, axis=2)

#     return float(np.sum(U ** 2 * distances_sq))


# def penalty(U):
#     return float(np.sum(U * (1.0 - U)))


# def penalized_objective(X, U, V, t):
#     return mssc_objective(X, U, V) + t * penalty(U)


# # ============================================================
# # 4. Choose a conservative rho using the Hessian condition
# #    from the convexity analysis around equations (23)-(24).
# #
# #    This is a conservative implementation bound, not a
# #    verbatim transcription of the paper's equation (24).
# # ============================================================

# def choose_rho(X, radius):
#     n = len(X)

#     xi = radius + np.max(np.linalg.norm(X, axis=1))
#     a = xi ** 2

#     # Sufficient bound derived from the block Hessian:
#     # (rho - 2*a) * (rho/n - 2) >= 16*a
#     rho = a + n + np.sqrt((a + n) ** 2 + 12.0 * a * n)

#     # Small numerical safety margin.
#     return float(rho * (1.0 + 1e-10) + 1e-12)


# # ============================================================
# # 5. IP-DCA
# #
# # Paper: equations (19)-(28), Section 3.
# #
# # U: shape (K, n), membership variables
# # V: shape (K, d), cluster centers
# # ============================================================

# def ip_dca(
#     X,
#     n_clusters,
#     t=10.0,
#     max_iter=1000,
#     tol=1e-6,
#     random_state=42
# ):
#     X = np.asarray(X, dtype=float)

#     if X.ndim != 2 or len(X) == 0:
#         raise ValueError("X must be a non-empty 2D array.")

#     if not np.all(np.isfinite(X)):
#         raise ValueError("X must contain only finite values.")

#     n, d = X.shape
#     K = n_clusters

#     if not isinstance(K, (int, np.integer)) or not 1 <= K <= n:
#         raise ValueError("n_clusters must be an integer in [1, n].")

#     if t <= 0:
#         raise ValueError("t must be positive.")

#     rng = np.random.default_rng(random_state)

#     # --------------------------------------------------------
#     # A. Feasible set:
#     # U columns belong to the simplex.
#     # Each center belongs to a ball of radius r.
#     #
#     # Paper, Section 3.1:
#     # r^2 = sum_k ||x_k||^2
#     # --------------------------------------------------------

#     radius = float(np.sqrt(np.sum(X ** 2)))
#     rho = choose_rho(X, radius)

#     # Initial memberships: random one-hot assignments.
#     initial_labels = rng.integers(0, K, size=n)
#     U = np.zeros((K, n), dtype=float)
#     U[initial_labels, np.arange(n)] = 1.0

#     # Initialize centers from assigned points.
#     V = np.zeros((K, d), dtype=float)

#     for i in range(K):
#         members = X[initial_labels == i]

#         if len(members) > 0:
#             V[i] = members.mean(axis=0)
#         else:
#             V[i] = X[rng.integers(n)]

#         V[i] = project_ball(V[i], radius)

#     history = []

#     # --------------------------------------------------------
#     # B. DCA iterations
#     # Equations (25)-(28) in the paper.
#     # --------------------------------------------------------

#     for iteration in range(max_iter):

#         U_old = U.copy()
#         V_old = V.copy()

#         # Squared distances:
#         # D[i,k] = ||x_k - v_i||^2
#         diff = X[None, :, :] - V[:, None, :]
#         D = np.sum(diff ** 2, axis=2)

#         # Equation (27): gradient with respect to U.
#         #
#         # Y[i,k] = (rho*u[i,k] - 2*u[i,k]*D[i,k]
#         #           + 2*t*u[i,k] - t)
#         Y = (
#             rho * U
#             - 2.0 * U * D
#             + 2.0 * t * U
#             - t
#         )

#         # Equation (27): gradient with respect to V.
#         #
#         # Z[i] = rho*v_i
#         #        - 2*sum_k (v_i - x_k)*u[i,k]^2
#         Z = (
#             rho * V
#             - 2.0 * np.sum(
#                 (V[:, None, :] - X[None, :, :])
#                 * (U[:, :, None] ** 2),
#                 axis=1
#             )
#         )

#         # Equation (28):
#         # Project each membership column onto the simplex.
#         U = np.column_stack([
#             project_simplex(Y[:, k] / rho)
#             for k in range(n)
#         ])

#         # Equation (28):
#         # Project each center onto its Euclidean ball.
#         V = np.vstack([
#             project_ball(Z[i] / rho, radius)
#             for i in range(K)
#         ])

#         # Record the objective.
#         value = penalized_objective(X, U, V, t)
#         history.append(value)

#         # Stopping condition based on the change in (U,V).
#         change = np.sqrt(
#             np.sum((U - U_old) ** 2)
#             + np.sum((V - V_old) ** 2)
#         )

#         if change <= tol:
#             break

#     # Decode cluster labels from memberships.
#     labels = np.argmax(U, axis=0)

#     return {
#         "labels": labels,
#         "memberships": U,
#         "centers": V,
#         "rho": rho,
#         "radius": radius,
#         "iterations": len(history),
#         "objective_history": history,
#         "mssc": mssc_objective(X, U, V),
#         "penalty": penalty(U),
#         "penalized_objective": penalized_objective(X, U, V, t)
#     }


#
"""
IP-DCA: DCA cho bài toán phân cụm MSSC dạng nguyên-hỗn hợp (IP-MSSC).

Theo bài báo: Le Thi Hoai An, Le Hoai Minh, Pham Dinh Tao,
"New and efficient DCA based algorithms for minimum sum-of-squares clustering",
Pattern Recognition 47 (2014) 388-401, mục 3.

Công thức dùng:
  (20)  min F1(U,V) = F(U,V) + t p(U),  (U,V) in Delta^n x C
        F(U,V) = sum_k sum_i u_ik^2 ||x_k - v_i||^2
        p(U)   = sum_i sum_k u_ik (1 - u_ik)
  (24)  cận dưới của rho để H lồi
  (27)  gradient (Y, Z) của H1
  (28)  U_k^{l+1} = Proj_Delta(Y_k / rho),  V_i^{l+1} = Proj_ball(Z_i / rho)
"""
import numpy as np


def _sq_dists(V, X):
    """D[i, k] = ||x_k - v_i||^2, kích thước (c, n)."""
    return ((V[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)


def _proj_simplex(Y):
    """Chiếu từng cột của Y (c x n) lên đơn hình {y >= 0, sum y = 1}."""
    c, n = Y.shape
    S = -np.sort(-Y, axis=0)
    css = np.cumsum(S, axis=0) - 1.0
    idx = np.arange(1, c + 1)[:, None]
    k = ((S - css / idx) > 0).sum(axis=0)
    theta = css[k - 1, np.arange(n)] / k
    return np.maximum(Y - theta, 0.0)


def _proj_ball(Z, r):
    """Chiếu từng hàng của Z (c x d) lên hình cầu tâm 0, bán kính r."""
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z * np.where(norms <= r, 1.0, r / np.maximum(norms, 1e-300))


def mssc_objective(X, U, V):
    """F(U, V) = sum_k sum_i u_ik^2 ||x_k - v_i||^2."""
    return float((U ** 2 * _sq_dists(V, X)).sum())


def penalty(U):
    """p(U) = sum_i sum_k u_ik (1 - u_ik); bằng 0 khi U là ma trận 0/1."""
    return float((U * (1.0 - U)).sum())


def penalized_objective(X, U, V, t):
    """F1(U, V) = F(U, V) + t * p(U)   (công thức (20))."""
    return mssc_objective(X, U, V) + t * penalty(U)


def _rho_bound(X, r):
    """Công thức (24): rho >= n [ (xi^2/n + 1) + sqrt((xi^2/n + 1)^2 + 12 xi^2 / n) ]."""
    n = X.shape[0]
    xi = r + np.linalg.norm(X, axis=1).max()
    a = xi ** 2 / n + 1.0
    return n * (a + np.sqrt(a ** 2 + 12.0 * xi ** 2 / n))


def _step(U, V, X, rho, t, r):
    """Một bước DCA: tính (Y, Z) theo (27) rồi chiếu theo (28)."""
    D = _sq_dists(V, X)
    Y = rho * U - 2.0 * U * D + 2.0 * t * U - t
    U2 = U ** 2
    Z = rho * V - 2.0 * (V * U2.sum(axis=1, keepdims=True) - U2 @ X)
    return _proj_simplex(Y / rho), _proj_ball(Z / rho, r)


def ip_dca(X, n_clusters, c_init = None, t=10.0, max_iter=1000, tol=1e-6, random_state=42):
    """
    Phân cụm MSSC bằng IP-DCA.

    Tham số
    -------
    X            : (n, d) dữ liệu.
    n_clusters   : số cụm c.
    t            : tham số phạt của p(U) trong (20).
    max_iter     : số vòng lặp DCA tối đa.
    tol          : dừng khi ||(U, V)^{l+1} - (U, V)^l|| <= tol.
    random_state : seed; tâm cụm khởi tạo là c điểm lấy ngẫu nhiên trong X.

    Trả về (dict)
    -------------
    labels              : (n,) nhãn cụm.
    memberships         : (c, n) ma trận thành viên U (xấp xỉ 0/1).
    centers             : (c, d) tâm cụm V.
    rho                 : giá trị rho cuối cùng.
    radius              : bán kính r của hình cầu chứa tâm cụm.
    iterations          : len(history).
    objective_history   : giá trị F1 qua các vòng lặp.
    mssc                : F(U, V).
    penalty             : p(U).
    penalized_objective : F1(U, V) = F + t * p.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    c = n_clusters
    if not 2 <= c <= n:
        raise ValueError("n_clusters phải thỏa 2 <= n_clusters <= n")

    rng = np.random.default_rng(random_state)

    # r^2 = sum_k ||x_k||^2: bán kính hình cầu chứa các tâm tối ưu
    r = np.sqrt((X ** 2).sum())

    # khởi tạo: tâm = c điểm ngẫu nhiên của dữ liệu, U = gán cứng theo tâm gần nhất
    V = X[rng.choice(n, size=c, replace=False)].copy() if c_init is None else c_init
    U = np.zeros((c, n))
    U[_sq_dists(V, X).argmin(axis=0), np.arange(n)] = 1.0

    # rho khởi tạo theo (24), rồi giảm dần (thủ tục "Update rho", mục 2.3)
    rho_max = _rho_bound(X, r)
    rho, tau = rho_max, 0.9
    history = [penalized_objective(X, U, V, t)]
    warmup = 0
    for _ in range(min(200, max_iter)):
        Un, Vn = _step(U, V, X, tau * rho, t, r)
        f_new = penalized_objective(X, Un, Vn, t)
        if f_new > history[-1]:
            break
        U, V, rho = Un, Vn, tau * rho
        history.append(f_new)
        warmup += 1

    n_iter = warmup
    for n_iter in range(warmup + 1, max_iter + 1):
        Un, Vn = _step(U, V, X, rho, t, r)
        # nếu F1 tăng thì rho quá nhỏ (H1 mất tính lồi): tăng rho rồi thử lại
        tries = 0
        while penalized_objective(X, Un, Vn, t) > history[-1] + 1e-12 and rho < rho_max and tries < 100:
            rho = min(rho / tau, rho_max)
            Un, Vn = _step(U, V, X, rho, t, r)
            tries += 1
        diff = np.sqrt(((Un - U) ** 2).sum() + ((Vn - V) ** 2).sum())
        U, V = Un, Vn
        history.append(penalized_objective(X, U, V, t))
        if diff <= tol:
            break

    labels = U.argmax(axis=0)
    return {
        "labels": labels,
        "memberships": U,
        "centers": V,
        "rho": rho,
        "radius": r,
        "iterations": len(history),
        "objective_history": history,
        "mssc": mssc_objective(X, U, V),
        "penalty": penalty(U),
        "penalized_objective": penalized_objective(X, U, V, t),
    }


# if __name__ == "__main__":
#     rng = np.random.default_rng(0)
#     centers = np.array([[0, 0], [6, 6], [0, 7]], float)
#     X = np.vstack([rng.normal(m, 0.8, (100, 2)) for m in centers])
#     res = ip_dca(X, 3)
#     labels, V = res["labels"], res["centers"]
#     print("iterations:", res["iterations"], "| rho: %.4g | radius: %.4g" % (res["rho"], res["radius"]))
#     print("mssc: %.4f | penalty: %.2e | penalized: %.4f" % (res["mssc"], res["penalty"], res["penalized_objective"]))
#     print("SSE (theo labels):", round(((X - V[labels]) ** 2).sum(), 4))
#     print("kích thước cụm:", np.bincount(labels, minlength=3))