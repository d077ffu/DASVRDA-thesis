import time
import csv
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import expit
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42

PILOT = True

if PILOT:
    BATCH_SIZES = [128, 512, 2048]
    MAX_DCA_OUTER = 8
else:
    BATCH_SIZES = [32, 128, 512, 2048]
    MAX_DCA_OUTER = 20

CAP_LAMBDA = 1e-4
THETA_CAP = 1e-2

TOL = 1e-3
CONSECUTIVE_CHECKS = 2

INNER_STAGES = 1
SGD_INNER_EPOCHS = 3

SGD_LR_SCALE = 2.0
SVRG_LR_SCALE = 10.0
DAS_STEP_SCALE = 20.0


def load_covtype():
    print("Downloading/loading Covertype ...")
    X, y_raw = fetch_covtype(return_X_y=True)

    y = np.where(y_raw == 2, 1.0, -1.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)

    return X_train, y_train, X_test, y_test


def logistic_loss(X, y, w):
    margin = y * (X @ w)
    return float(np.mean(np.logaddexp(0.0, -margin)))


def nonconvex_objective(X, y, w, lam, theta):
    penalty = lam * np.minimum(np.abs(w), theta).sum()
    return logistic_loss(X, y, w) + float(penalty)


def full_logistic_gradient(X, y, w):
    margin = y * (X @ w)
    coeff = -y * expit(-margin)
    return np.asarray(X.T @ coeff).ravel() / X.shape[0]


def batch_logistic_gradient(Xb, yb, w):
    margin = yb * (Xb @ w)
    coeff = -yb * expit(-margin)
    return np.asarray(Xb.T @ coeff).ravel() / Xb.shape[0]


def accuracy(X, y, w):
    score = X @ w
    pred = np.where(score >= 0.0, 1.0, -1.0)
    return float(np.mean(pred == y))


def capped_h_subgradient(w, lam, theta):
    u = np.zeros_like(w)
    mask = np.abs(w) > theta
    u[mask] = lam * np.sign(w[mask])
    return u


def soft_threshold(v, tau):
    return np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)


def prox_sgd(X, y, w0, u, lam, batch_size, lr, epochs, seed):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    w = w0.copy()
    grad_count = 0

    for _ in range(epochs):
        perm = rng.permutation(n)

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]

            g = batch_logistic_gradient(
                X[idx], y[idx], w
            ) - u

            w = soft_threshold(
                w - lr * g,
                lr * lam
            )

            grad_count += len(idx)

    return w, grad_count


def prox_svrg(X, y, w0, u, lam, batch_size, lr, stages, seed):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    w = w0.copy()
    grad_count = 0

    for _ in range(stages):
        snapshot = w.copy()

        full_grad = full_logistic_gradient(
            X, y, snapshot
        ) - u

        grad_count += n

        perm = rng.permutation(n)

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]

            g_current = batch_logistic_gradient(
                X[idx], y[idx], w
            )

            g_snapshot = batch_logistic_gradient(
                X[idx], y[idx], snapshot
            )

            v = g_current - g_snapshot + full_grad

            w = soft_threshold(
                w - lr * v,
                lr * lam
            )

            grad_count += 2 * len(idx)

    return w, grad_count


def estimate_component_L(X):
    row_sq_norm = np.sum(
        X.astype(np.float64) ** 2,
        axis=1
    )
    max_sq_norm = float(np.max(row_sq_norm))
    L = 0.25 * max_sq_norm
    return L, max_sq_norm


def prox_dasvrda(X, y, w0, u, lam, batch_size, stages, step_scale, seed, L):
    rng = np.random.default_rng(seed)

    n, d = X.shape
    m_inner = int(np.ceil(n / batch_size))

    gamma = (
        3.0
        + np.sqrt(9.0 + 8.0 / (m_inner + 1.0))
    ) / 2.0

    base_step = 1.0 / (
        (1.0 + gamma * (m_inner + 1.0)) * L
    )

    step_size = step_scale * base_step

    w = w0.copy()
    wold = w0.copy()
    wold_ = w0.copy()
    z = w0.copy()

    grad_count = 0

    for k in range(stages):
        thetas = (
            (1.0 - 1.0 / gamma)
            * (k + 3.0)
            / 2.0
        )

        thetas_old = (
            (1.0 - 1.0 / gamma)
            * (k + 2.0)
            / 2.0
        )

        weight1 = (thetas_old - 1.0) / thetas
        weight2 = thetas_old / thetas

        w = (
            wold
            + weight1 * (wold - wold_)
            + weight2 * (z - wold)
        )

        gold = full_logistic_gradient(
            X, y, wold
        ) - u

        grad_count += n

        avg = np.zeros(d, dtype=w.dtype)
        z_inner = w.copy()
        z0 = w.copy()

        perm = rng.permutation(n)
        inner_id = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]

            theta = 2.0 / (inner_id + 2.0)
            theta_old = 2.0 / (inner_id + 1.0)

            y_point = (
                (1.0 - theta) * w
                + theta * z_inner
            )

            g_current = batch_logistic_gradient(
                X[idx], y[idx], y_point
            )

            g_snapshot = batch_logistic_gradient(
                X[idx], y[idx], wold
            )

            g = gold + g_current - g_snapshot

            avg = (
                (1.0 - theta) * avg
                + theta * g
            )

            coeff = step_size / (
                theta * theta_old
            )

            z_inner = soft_threshold(
                z0 - coeff * avg,
                coeff * lam
            )

            w = (
                (1.0 - theta) * w
                + theta * z_inner
            )

            grad_count += 2 * len(idx)
            inner_id += 1

        wold_ = wold.copy()
        wold = w.copy()
        z = z_inner.copy()

    return w, grad_count


def run_dca(
    method,
    X_train, y_train,
    X_test, y_test,
    batch_size,
    L,
    lr_sgd,
    lr_svrg
):
    w = np.zeros(
        X_train.shape[1],
        dtype=np.float32
    )

    prev_obj = nonconvex_objective(
        X_train, y_train, w,
        CAP_LAMBDA, THETA_CAP
    )

    stable_count = 0
    converged = False
    grad_count = 0

    t0 = time.perf_counter()

    for outer in range(MAX_DCA_OUTER):
        u = capped_h_subgradient(
            w, CAP_LAMBDA, THETA_CAP
        )

        if method == "SGD":
            w, used = prox_sgd(
                X_train, y_train,
                w, u, CAP_LAMBDA,
                batch_size, lr_sgd,
                SGD_INNER_EPOCHS,
                SEED + outer
            )

        elif method == "SVRG":
            w, used = prox_svrg(
                X_train, y_train,
                w, u, CAP_LAMBDA,
                batch_size, lr_svrg,
                INNER_STAGES,
                SEED + outer
            )

        elif method == "DASVRDA":
            w, used = prox_dasvrda(
                X_train, y_train,
                w, u, CAP_LAMBDA,
                batch_size,
                INNER_STAGES,
                DAS_STEP_SCALE,
                SEED + outer,
                L
            )

        grad_count += used

        obj = nonconvex_objective(
            X_train, y_train, w,
            CAP_LAMBDA, THETA_CAP
        )

        rel_change = (
            abs(obj - prev_obj)
            / max(abs(prev_obj), 1e-12)
        )

        elapsed = time.perf_counter() - t0

        print(
            f"{method:8s} | batch={batch_size:4d} | "
            f"DCA={outer+1:02d} | "
            f"Obj={obj:.8f} | "
            f"RelChange={rel_change:.3e} | "
            f"Time={elapsed:.2f}s"
        )

        if rel_change < TOL:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= CONSECUTIVE_CHECKS:
            converged = True
            break

        prev_obj = obj

    total_time = time.perf_counter() - t0
    test_acc = accuracy(X_test, y_test, w)

    return {
        "method": method,
        "batch_size": batch_size,
        "converged": converged,
        "outer_iters": outer + 1,
        "time": total_time,
        "accuracy": test_acc,
        "objective": obj,
        "grad_evals": grad_count
    }


def main():
    X_train, y_train, X_test, y_test = load_covtype()

    print()
    print("Train:", X_train.shape)
    print("Test :", X_test.shape)
    print("Positive ratio train:", np.mean(y_train == 1.0))
    print("Positive ratio test :", np.mean(y_test == 1.0))

    L, max_sq_norm = estimate_component_L(X_train)

    lr_sgd = SGD_LR_SCALE / L
    lr_svrg = SVRG_LR_SCALE / L

    print()
    print(f"max ||x_i||^2 = {max_sq_norm:.6f}")
    print(f"L estimate     = {L:.6f}")
    print(f"SGD lr         = {lr_sgd:.6e}")
    print(f"SVRG lr        = {lr_svrg:.6e}")
    print(f"DAS scale      = {DAS_STEP_SCALE}")
    print(f"Batch sizes    = {BATCH_SIZES}")
    print()

    methods = ["SGD", "SVRG", "DASVRDA"]
    results = []

    for batch_size in BATCH_SIZES:
        print()
        print("#" * 90)
        print(f"BATCH SIZE = {batch_size}")
        print("#" * 90)

        for method in methods:
            result = run_dca(
                method,
                X_train, y_train,
                X_test, y_test,
                batch_size,
                L,
                lr_sgd,
                lr_svrg
            )

            results.append(result)

    print()
    print("=" * 110)
    print("FINAL SUMMARY")
    print("=" * 110)

    for r in results:
        status = "YES" if r["converged"] else "NO"

        print(
            f"{r['method']:8s} | "
            f"batch={r['batch_size']:4d} | "
            f"Converged={status:3s} | "
            f"Outer={r['outer_iters']:2d} | "
            f"Time={r['time']:.2f}s | "
            f"Acc={100*r['accuracy']:.3f}% | "
            f"Obj={r['objective']:.8f}"
        )

    with open(
        "covtype_dca_batch_results.csv",
        "w",
        newline=""
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "Method",
            "Batch Size",
            "Converged",
            "DCA Outer Iterations",
            "Time [s]",
            "Test Accuracy",
            "Final Nonconvex Objective",
            "Component Gradient Evaluations"
        ])

        for r in results:
            writer.writerow([
                r["method"],
                r["batch_size"],
                r["converged"],
                r["outer_iters"],
                r["time"],
                r["accuracy"],
                r["objective"],
                r["grad_evals"]
            ])

    plt.figure(figsize=(9, 6))

    for method in methods:
        rr = [
            r for r in results
            if r["method"] == method
        ]

        plt.plot(
            [r["batch_size"] for r in rr],
            [r["time"] for r in rr],
            marker="o",
            label=method
        )

    plt.xscale("log", base=2)
    plt.xlabel("Mini-batch Size")
    plt.ylabel("Elapsed Time [s]")
    plt.title("Covertype + DCA: Runtime vs Mini-batch Size")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "covtype_dca_batch_runtime.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.figure(figsize=(9, 6))

    for method in methods:
        rr = [
            r for r in results
            if r["method"] == method
        ]

        plt.plot(
            [r["batch_size"] for r in rr],
            [100.0 * r["accuracy"] for r in rr],
            marker="o",
            label=method
        )

    plt.xscale("log", base=2)
    plt.xlabel("Mini-batch Size")
    plt.ylabel("Test Accuracy [%]")
    plt.title("Covertype + DCA: Accuracy vs Mini-batch Size")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "covtype_dca_batch_accuracy.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":
    main()
