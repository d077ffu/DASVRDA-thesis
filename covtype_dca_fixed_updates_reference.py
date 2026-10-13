import csv
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import expit
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
BATCH_SIZES = [128, 512, 2048]
METHODS = ["SGD", "SVRG", "DASVRDA"]

CAP_LAMBDA = 1e-4
THETA_CAP = 1e-2

# Same number of mini-batch updates for every batch size
INNER_UPDATES = 500
MAX_DCA_OUTER = 20

# Reference neighborhood
REF_TOL = 1e-3

# Lightweight tuning
TUNE_DCA_OUTER = 2
TUNE_INNER_UPDATES = 150

SGD_SCALE_CANDIDATES = [1.0, 4.0, 16.0, 64.0, 256.0]
SVRG_SCALE_CANDIDATES = [4.0, 16.0, 64.0, 256.0, 1024.0]
DAS_SCALE_CANDIDATES = [256.0, 1024.0, 4096.0, 16384.0, 65536.0]


def load_covtype():
    print("Loading Covertype ...")
    X, y_raw = fetch_covtype(return_X_y=True)

    y = np.where(y_raw == 2, 1.0, -1.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=SEED,
        stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (
        np.asarray(X_train, dtype=np.float32),
        np.asarray(y_train, dtype=np.float32),
        np.asarray(X_test, dtype=np.float32),
        np.asarray(y_test, dtype=np.float32),
    )


def logistic_loss(X, y, w):
    margin = y * (X @ w)
    return float(np.mean(np.logaddexp(0.0, -margin)))


def nonconvex_objective(X, y, w, lam, theta_cap):
    penalty = lam * np.minimum(np.abs(w), theta_cap).sum()
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
    pred = np.where(X @ w >= 0.0, 1.0, -1.0)
    return float(np.mean(pred == y))


def estimate_component_L(X):
    row_sq_norm = np.sum(X.astype(np.float64) ** 2, axis=1)
    max_sq_norm = float(np.max(row_sq_norm))
    return 0.25 * max_sq_norm, max_sq_norm


def capped_h_subgradient(w, lam, theta_cap):
    u = np.zeros_like(w)
    mask = np.abs(w) > theta_cap
    u[mask] = lam * np.sign(w[mask])
    return u


def soft_threshold(v, tau):
    return np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)


def random_batch(rng, n, batch_size):
    return rng.integers(0, n, size=batch_size)


def prox_sgd_fixed_updates(X, y, w0, u, lam, batch_size, lr, num_updates, seed):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    w = w0.copy()
    opt_time = 0.0
    grad_count = 0

    for _ in range(num_updates):
        idx = random_batch(rng, n, batch_size)

        t0 = time.perf_counter()
        g = batch_logistic_gradient(X[idx], y[idx], w) - u
        w = soft_threshold(w - lr * g, lr * lam)
        opt_time += time.perf_counter() - t0

        grad_count += batch_size

    return w, opt_time, grad_count


def prox_svrg_fixed_updates(X, y, w0, u, lam, batch_size, lr, num_updates, seed):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    w = w0.copy()
    snapshot = w0.copy()
    opt_time = 0.0
    grad_count = 0

    t0 = time.perf_counter()
    full_grad = full_logistic_gradient(X, y, snapshot) - u
    opt_time += time.perf_counter() - t0
    grad_count += n

    for _ in range(num_updates):
        idx = random_batch(rng, n, batch_size)

        t0 = time.perf_counter()
        g_current = batch_logistic_gradient(X[idx], y[idx], w)
        g_snapshot = batch_logistic_gradient(X[idx], y[idx], snapshot)
        v = g_current - g_snapshot + full_grad
        w = soft_threshold(w - lr * v, lr * lam)
        opt_time += time.perf_counter() - t0

        grad_count += 2 * batch_size

    return w, opt_time, grad_count


def prox_dasvrda_fixed_updates(
    X, y, w0, u, lam, batch_size,
    step_scale, num_updates, seed, L
):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    m_inner = num_updates

    gamma = (
        3.0 + np.sqrt(9.0 + 8.0 / (m_inner + 1.0))
    ) / 2.0

    base_step = 1.0 / (
        (1.0 + gamma * (m_inner + 1.0)) * L
    )
    step_size = step_scale * base_step

    w = w0.copy()
    snapshot = w0.copy()
    z_inner = w0.copy()
    z0 = w0.copy()
    avg = np.zeros(d, dtype=w.dtype)

    opt_time = 0.0
    grad_count = 0

    t0 = time.perf_counter()
    gold = full_logistic_gradient(X, y, snapshot) - u
    opt_time += time.perf_counter() - t0
    grad_count += n

    for inner_id in range(num_updates):
        idx = random_batch(rng, n, batch_size)

        theta = 2.0 / (inner_id + 2.0)
        theta_old = 2.0 / (inner_id + 1.0)

        t0 = time.perf_counter()

        y_point = (1.0 - theta) * w + theta * z_inner
        g_current = batch_logistic_gradient(X[idx], y[idx], y_point)
        g_snapshot = batch_logistic_gradient(X[idx], y[idx], snapshot)

        g = gold + g_current - g_snapshot
        avg = (1.0 - theta) * avg + theta * g

        coeff = step_size / (theta * theta_old)
        z_inner = soft_threshold(
            z0 - coeff * avg,
            coeff * lam
        )

        w = (1.0 - theta) * w + theta * z_inner

        opt_time += time.perf_counter() - t0
        grad_count += 2 * batch_size

    return w, opt_time, grad_count


def tune_parameter(method, candidates, X, y, batch_size, L):
    print()
    print(f"TUNING {method} | batch={batch_size}")

    best_param = None
    best_obj = np.inf

    for param in candidates:
        w = np.zeros(X.shape[1], dtype=np.float32)
        bad = False

        for outer in range(TUNE_DCA_OUTER):
            u = capped_h_subgradient(
                w, CAP_LAMBDA, THETA_CAP
            )

            if method == "SGD":
                lr = param / L
                w, _, _ = prox_sgd_fixed_updates(
                    X, y, w, u, CAP_LAMBDA,
                    batch_size, lr,
                    TUNE_INNER_UPDATES,
                    SEED + outer
                )

            elif method == "SVRG":
                lr = param / L
                w, _, _ = prox_svrg_fixed_updates(
                    X, y, w, u, CAP_LAMBDA,
                    batch_size, lr,
                    TUNE_INNER_UPDATES,
                    SEED + outer
                )

            elif method == "DASVRDA":
                w, _, _ = prox_dasvrda_fixed_updates(
                    X, y, w, u, CAP_LAMBDA,
                    batch_size, param,
                    TUNE_INNER_UPDATES,
                    SEED + outer,
                    L
                )

            obj = nonconvex_objective(
                X, y, w,
                CAP_LAMBDA, THETA_CAP
            )

            if not np.isfinite(obj) or obj > 1.5:
                bad = True
                break

        final_obj = np.inf if bad else obj

        print(
            f"  param={param:10.3f} | "
            f"objective={final_obj:.8f}"
        )

        if final_obj < best_obj:
            best_obj = final_obj
            best_param = param

    print(
        f"  BEST -> param={best_param} | "
        f"objective={best_obj:.8f}"
    )

    return best_param


def run_long_experiment(
    method, parameter,
    X_train, y_train,
    batch_size, L
):
    w = np.zeros(X_train.shape[1], dtype=np.float32)

    objective_history = []
    time_history = []
    update_history = []
    grad_history = []
    weight_history = []

    cumulative_opt_time = 0.0
    cumulative_updates = 0
    cumulative_grad = 0

    obj0 = nonconvex_objective(
        X_train, y_train, w,
        CAP_LAMBDA, THETA_CAP
    )

    objective_history.append(obj0)
    time_history.append(0.0)
    update_history.append(0)
    grad_history.append(0)
    weight_history.append(w.copy())

    print()
    print("=" * 88)
    print(
        f"{method} | batch={batch_size} | "
        f"parameter={parameter}"
    )
    print("=" * 88)

    for outer in range(MAX_DCA_OUTER):
        u = capped_h_subgradient(
            w, CAP_LAMBDA, THETA_CAP
        )

        if method == "SGD":
            lr = parameter / L
            w, used_time, used_grad = prox_sgd_fixed_updates(
                X_train, y_train, w, u, CAP_LAMBDA,
                batch_size, lr,
                INNER_UPDATES,
                SEED + outer
            )

        elif method == "SVRG":
            lr = parameter / L
            w, used_time, used_grad = prox_svrg_fixed_updates(
                X_train, y_train, w, u, CAP_LAMBDA,
                batch_size, lr,
                INNER_UPDATES,
                SEED + outer
            )

        elif method == "DASVRDA":
            w, used_time, used_grad = prox_dasvrda_fixed_updates(
                X_train, y_train, w, u, CAP_LAMBDA,
                batch_size, parameter,
                INNER_UPDATES,
                SEED + outer,
                L
            )

        cumulative_opt_time += used_time
        cumulative_updates += INNER_UPDATES
        cumulative_grad += used_grad

        # Full objective evaluation is intentionally excluded
        # from cumulative optimization time.
        obj = nonconvex_objective(
            X_train, y_train, w,
            CAP_LAMBDA, THETA_CAP
        )

        objective_history.append(obj)
        time_history.append(cumulative_opt_time)
        update_history.append(cumulative_updates)
        grad_history.append(cumulative_grad)
        weight_history.append(w.copy())

        print(
            f"DCA={outer+1:02d} | "
            f"updates={cumulative_updates:5d} | "
            f"Obj={obj:.8f} | "
            f"OptTime={cumulative_opt_time:.3f}s"
        )

        if not np.isfinite(obj):
            break

    return {
        "method": method,
        "batch_size": batch_size,
        "parameter": parameter,
        "objective": np.array(objective_history),
        "time": np.array(time_history),
        "updates": np.array(update_history),
        "grad": np.array(grad_history),
        "weights": weight_history,
    }


def main():
    X_train, y_train, X_test, y_test = load_covtype()

    print()
    print("Train shape :", X_train.shape)
    print("Test shape  :", X_test.shape)
    print("Batch sizes :", BATCH_SIZES)
    print("Updates / DCA outer :", INNER_UPDATES)
    print("Max DCA outer       :", MAX_DCA_OUTER)

    L, max_sq_norm = estimate_component_L(X_train)

    print(f"max ||x_i||^2 = {max_sq_norm:.6f}")
    print(f"L estimate     = {L:.6f}")

    # 1. Tune each method for each batch size
    selected = {}

    for batch_size in BATCH_SIZES:
        selected[("SGD", batch_size)] = tune_parameter(
            "SGD",
            SGD_SCALE_CANDIDATES,
            X_train, y_train,
            batch_size, L
        )

        selected[("SVRG", batch_size)] = tune_parameter(
            "SVRG",
            SVRG_SCALE_CANDIDATES,
            X_train, y_train,
            batch_size, L
        )

        selected[("DASVRDA", batch_size)] = tune_parameter(
            "DASVRDA",
            DAS_SCALE_CANDIDATES,
            X_train, y_train,
            batch_size, L
        )

    print()
    print("=" * 90)
    print("SELECTED PARAMETERS")
    print("=" * 90)

    for batch_size in BATCH_SIZES:
        print(
            f"batch={batch_size:4d} | "
            f"SGD={selected[('SGD', batch_size)]} | "
            f"SVRG={selected[('SVRG', batch_size)]} | "
            f"DAS={selected[('DASVRDA', batch_size)]}"
        )

    # 2. Long measured runs
    results = []

    for batch_size in BATCH_SIZES:
        for method in METHODS:
            results.append(
                run_long_experiment(
                    method,
                    selected[(method, batch_size)],
                    X_train, y_train,
                    batch_size, L
                )
            )

    # 3. Reference value = best objective observed across all runs
    all_objectives = np.concatenate([
        r["objective"][np.isfinite(r["objective"])]
        for r in results
    ])

    f_ref = float(np.min(all_objectives))
    target = f_ref + REF_TOL

    print()
    print("=" * 90)
    print("REFERENCE")
    print("=" * 90)
    print(f"F_ref  = {f_ref:.10f}")
    print(f"Target = {target:.10f}")

    # 4. First checkpoint that reaches reference neighborhood
    summary = []

    for r in results:
        idx = np.where(
            r["objective"] <= target
        )[0]

        if len(idx) == 0:
            reached = False
            first_idx = len(r["objective"]) - 1
        else:
            reached = True
            first_idx = int(idx[0])

        w_at_stop = r["weights"][first_idx]

        acc_at_stop = accuracy(
            X_test, y_test, w_at_stop
        )

        summary.append({
            "method": r["method"],
            "batch_size": r["batch_size"],
            "parameter": r["parameter"],
            "reached": reached,
            "time": float(r["time"][first_idx]),
            "updates": int(r["updates"][first_idx]),
            "grad_evals": int(r["grad"][first_idx]),
            "objective": float(r["objective"][first_idx]),
            "accuracy": acc_at_stop,
        })

    print()
    print("=" * 118)
    print("FINAL SUMMARY")
    print("=" * 118)

    for s in summary:
        status = "REACHED" if s["reached"] else "NOT REACHED"

        print(
            f"{s['method']:8s} | "
            f"batch={s['batch_size']:4d} | "
            f"{status:11s} | "
            f"Time={s['time']:.3f}s | "
            f"Updates={s['updates']:5d} | "
            f"Acc={100*s['accuracy']:.3f}% | "
            f"Obj={s['objective']:.8f}"
        )

    # 5. Save CSV
    with open(
        "covtype_dca_fixed_updates_summary.csv",
        "w",
        newline=""
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "Method",
            "Batch Size",
            "Selected Parameter",
            "F_ref",
            "Reference Tolerance",
            "Reached Reference Neighborhood",
            "Time to Reference [s]",
            "Updates to Reference",
            "Component Gradient Evaluations",
            "Objective at Stop",
            "Test Accuracy at Stop",
        ])

        for s in summary:
            writer.writerow([
                s["method"],
                s["batch_size"],
                s["parameter"],
                f_ref,
                REF_TOL,
                s["reached"],
                s["time"],
                s["updates"],
                s["grad_evals"],
                s["objective"],
                s["accuracy"],
            ])

    # 6. Time to reference vs batch size
    plt.figure(figsize=(9, 6))

    for method in METHODS:
        ss = [s for s in summary if s["method"] == method]

        plt.plot(
            [s["batch_size"] for s in ss],
            [
                s["time"] if s["reached"] else np.nan
                for s in ss
            ],
            marker="o",
            label=method
        )

    plt.xscale("log", base=2)
    plt.xlabel("Mini-batch Size")
    plt.ylabel("Time to Reference [s]")
    plt.title("Covertype + DCA: Time to Reference vs Mini-batch Size")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "covtype_dca_fixed_updates_time_to_ref.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    # 7. Accuracy at reference vs batch size
    plt.figure(figsize=(9, 6))

    for method in METHODS:
        ss = [s for s in summary if s["method"] == method]

        plt.plot(
            [s["batch_size"] for s in ss],
            [100.0 * s["accuracy"] for s in ss],
            marker="o",
            label=method
        )

    plt.xscale("log", base=2)
    plt.xlabel("Mini-batch Size")
    plt.ylabel("Test Accuracy [%]")
    plt.title("Covertype + DCA: Accuracy at Reference")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "covtype_dca_fixed_updates_accuracy.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    # 8. Requested graph: time vs accuracy
    plt.figure(figsize=(9, 6))

    marker_map = {
        "SGD": "o",
        "SVRG": "s",
        "DASVRDA": "^",
    }

    for method in METHODS:
        ss = [
            s for s in summary
            if s["method"] == method and s["reached"]
        ]

        if not ss:
            continue

        x = [s["time"] for s in ss]
        y = [100.0 * s["accuracy"] for s in ss]

        plt.scatter(
            x, y,
            marker=marker_map[method],
            s=80,
            label=method
        )

        for s in ss:
            plt.annotate(
                f"b={s['batch_size']}",
                (s["time"], 100.0 * s["accuracy"]),
                xytext=(6, 6),
                textcoords="offset points"
            )

    plt.xlabel("Time to Reference [s]")
    plt.ylabel("Test Accuracy [%]")
    plt.title("Covertype + DCA: Speed-Accuracy Trade-off")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "covtype_dca_fixed_updates_time_accuracy.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    # 9. Objective vs time, one figure per batch size
    for batch_size in BATCH_SIZES:
        plt.figure(figsize=(9, 6))

        for method in METHODS:
            r = next(
                item for item in results
                if item["method"] == method
                and item["batch_size"] == batch_size
            )

            plt.plot(
                r["time"],
                r["objective"],
                marker="o",
                label=method
            )

        plt.axhline(
            target,
            linestyle="--",
            label="Reference target"
        )

        plt.xlabel("Optimization Time [s]")
        plt.ylabel("Nonconvex Objective")
        plt.title(
            f"Covertype + DCA: Objective vs Time (batch={batch_size})"
        )
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            f"covtype_dca_fixed_updates_objective_batch{batch_size}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


if __name__ == "__main__":
    main()
