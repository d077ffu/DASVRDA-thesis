import time
import csv
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import expit
from sklearn.datasets import load_svmlight_file

# ============================================================
# RCV1 + capped-L1 + DCA
# Compare:
#   DCA + Prox-SGD
#   DCA + Prox-SVRG
#   DCA + Prox-DASVRDA
# ============================================================

TRAIN_FILE = "rcv1_train.binary"
TEST_FILE = "rcv1_test.binary"

SEED = 42
BATCH_SIZE = 8

# capped-L1: lambda * sum_j min(|w_j|, theta_cap)
CAP_LAMBDA = 1e-4
THETA_CAP = 1e-2

# DCA outer iterations
DCA_OUTER = 8

# Fair inner budget per DCA iteration:
# SGD: 6 epochs = 6n gradient evaluations
# SVRG/DASVRDA: 2 stages x 3n = 6n gradient evaluations
INNER_STAGES = 2
SGD_INNER_EPOCHS = 3 * INNER_STAGES

# Starting values. Retune for capped-L1 before final thesis results.
SGD_LR = 1.0
SVRG_LR = 10.0
DAS_STEP_SCALE = 20.0

TEST_EVAL_SIZE = 10000
ZERO_TOL = 1e-10


def load_rcv1(train_file, test_file):
    X_train, y_train = load_svmlight_file(train_file)
    X_test, y_test = load_svmlight_file(test_file, n_features=X_train.shape[1])

    X_train = X_train.tocsr()
    X_test = X_test.tocsr()

    y_train = np.where(y_train > 0, 1.0, -1.0)
    y_test = np.where(y_test > 0, 1.0, -1.0)

    return X_train, y_train, X_test, y_test


def logistic_loss(X, y, w):
    margin = y * (X @ w)
    return float(np.mean(np.logaddexp(0.0, -margin)))


def capped_l1_penalty(w, lam, theta_cap):
    return float(lam * np.minimum(np.abs(w), theta_cap).sum())


def nonconvex_objective(X, y, w, lam, theta_cap):
    return logistic_loss(X, y, w) + capped_l1_penalty(w, lam, theta_cap)


def full_logistic_gradient(X, y, w):
    margin = y * (X @ w)
    coeff = -y * expit(-margin)
    grad = X.T @ coeff
    return np.asarray(grad).ravel() / X.shape[0]


def batch_logistic_gradient(Xb, yb, w):
    margin = yb * (Xb @ w)
    coeff = -yb * expit(-margin)
    grad = Xb.T @ coeff
    return np.asarray(grad).ravel() / Xb.shape[0]


def accuracy(X, y, w):
    score = X @ w
    pred = np.where(score >= 0.0, 1.0, -1.0)
    return float(np.mean(pred == y))


def sparsity(w, tol=ZERO_TOL):
    return float(np.mean(np.abs(w) <= tol))


# capped-L1 DC decomposition:
# lambda min(|w_j|, theta)
# = lambda |w_j| - lambda (|w_j|-theta)_+
def capped_h_subgradient(w, lam, theta_cap):
    u = np.zeros_like(w)
    mask = np.abs(w) > theta_cap
    u[mask] = lam * np.sign(w[mask])
    return u


def soft_threshold(v, tau):
    return np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)


def solve_subproblem_prox_sgd(
    X, y, w0, u, lam, batch_size, learning_rate, epochs, seed
):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    w = w0.copy()
    grad_count = 0

    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            Xb = X[idx]
            yb = y[idx]

            g = batch_logistic_gradient(Xb, yb, w) - u
            w = soft_threshold(w - learning_rate * g, learning_rate * lam)
            grad_count += len(idx)

    return w, grad_count


def solve_subproblem_prox_svrg(
    X, y, w0, u, lam, batch_size, learning_rate, stages, seed
):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    w = w0.copy()
    grad_count = 0

    for _ in range(stages):
        snapshot = w.copy()
        full_grad = full_logistic_gradient(X, y, snapshot) - u
        grad_count += n

        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            Xb = X[idx]
            yb = y[idx]

            g_current = batch_logistic_gradient(Xb, yb, w)
            g_snapshot = batch_logistic_gradient(Xb, yb, snapshot)
            v = g_current - g_snapshot + full_grad

            w = soft_threshold(w - learning_rate * v, learning_rate * lam)
            grad_count += 2 * len(idx)

    return w, grad_count


def estimate_component_L(X):
    row_sq_norm = np.asarray(X.multiply(X).sum(axis=1)).ravel()
    max_sq_norm = float(row_sq_norm.max())
    L = 0.25 * max_sq_norm
    return L, max_sq_norm


def solve_subproblem_prox_dasvrda(
    X, y, w0, u, lam, batch_size, stages, step_scale, seed, L
):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    m_inner = int(np.ceil(n / batch_size))

    gamma = (3.0 + np.sqrt(9.0 + 8.0 / (m_inner + 1.0))) / 2.0
    base_step = 1.0 / ((1.0 + gamma * (m_inner + 1.0)) * L)
    step_size = step_scale * base_step

    w = w0.copy()
    wold = w0.copy()
    wold_ = w0.copy()
    z = w0.copy()
    grad_count = 0

    for k in range(stages):
        thetas = (1.0 - 1.0 / gamma) * (k + 3.0) / 2.0
        thetas_old = (1.0 - 1.0 / gamma) * (k + 2.0) / 2.0
        weight1 = (thetas_old - 1.0) / thetas
        weight2 = thetas_old / thetas

        w = wold + weight1 * (wold - wold_) + weight2 * (z - wold)

        # Smooth part of the DCA subproblem: logistic(w) - <u,w>
        gold = full_logistic_gradient(X, y, wold) - u
        grad_count += n

        avg = np.zeros(d)
        z_inner = w.copy()
        z0 = w.copy()
        perm = rng.permutation(n)
        inner_id = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            Xb = X[idx]
            yb = y[idx]

            theta = 2.0 / (inner_id + 2.0)
            theta_old = 2.0 / (inner_id + 1.0)

            y_point = (1.0 - theta) * w + theta * z_inner

            g_current = batch_logistic_gradient(Xb, yb, y_point)
            g_snapshot = batch_logistic_gradient(Xb, yb, wold)
            g = gold + g_current - g_snapshot

            avg = (1.0 - theta) * avg + theta * g

            coeff = step_size / (theta * theta_old)
            z_inner = soft_threshold(
                z0 - coeff * avg,
                coeff * lam
            )

            w = (1.0 - theta) * w + theta * z_inner

            grad_count += 2 * len(idx)
            inner_id += 1

        wold_ = wold.copy()
        wold = w.copy()
        z = z_inner.copy()

    return w, grad_count, base_step, step_size


def run_dca(
    method,
    X_train,
    y_train,
    X_test_eval,
    y_test_eval,
    lam,
    theta_cap,
    dca_outer,
    batch_size,
    seed,
    sgd_lr,
    svrg_lr,
    das_step_scale,
    inner_stages,
    sgd_inner_epochs,
    L
):
    d = X_train.shape[1]
    w = np.zeros(d)

    obj_hist = []
    acc_hist = []
    sparsity_hist = []
    grad_hist = []
    time_hist = []

    total_grad = 0
    t0 = time.perf_counter()

    print("\n" + "=" * 84)
    print(f"DCA + {method}")
    print("=" * 84)

    initial_obj = nonconvex_objective(X_train, y_train, w, lam, theta_cap)
    print(f"Initial objective = {initial_obj:.10f} | sparsity = {sparsity(w):.4f}")

    for t in range(dca_outer):
        u = capped_h_subgradient(w, lam, theta_cap)

        if method == "Prox-SGD":
            w_new, used_grad = solve_subproblem_prox_sgd(
                X_train, y_train, w, u, lam, batch_size,
                sgd_lr, sgd_inner_epochs, seed + t
            )
            extra = ""

        elif method == "Prox-SVRG":
            w_new, used_grad = solve_subproblem_prox_svrg(
                X_train, y_train, w, u, lam, batch_size,
                svrg_lr, inner_stages, seed + t
            )
            extra = ""

        elif method == "Prox-DASVRDA":
            w_new, used_grad, base_step, actual_step = solve_subproblem_prox_dasvrda(
                X_train, y_train, w, u, lam, batch_size,
                inner_stages, das_step_scale, seed + t, L
            )
            extra = f" | base_step={base_step:.3e} | step={actual_step:.3e}"

        else:
            raise ValueError(f"Unknown method: {method}")

        w = w_new
        total_grad += used_grad

        obj = nonconvex_objective(X_train, y_train, w, lam, theta_cap)
        acc = accuracy(X_test_eval, y_test_eval, w)
        sp = sparsity(w)
        elapsed = time.perf_counter() - t0

        obj_hist.append(obj)
        acc_hist.append(acc)
        sparsity_hist.append(sp)
        grad_hist.append(total_grad)
        time_hist.append(elapsed)

        print(
            f"DCA {t+1:02d}/{dca_outer} | "
            f"Obj={obj:.10f} | Acc={acc:.4f} | "
            f"Sparsity={sp:.4f} | Grad={total_grad:,} | "
            f"Time={elapsed:.3f}s{extra}"
        )

    return {
        "method": method,
        "w": w,
        "objective": np.array(obj_hist),
        "accuracy": np.array(acc_hist),
        "sparsity": np.array(sparsity_hist),
        "grad": np.array(grad_hist),
        "time": np.array(time_hist)
    }


def main():
    print("Loading RCV1 ...")
    X_train, y_train, X_test, y_test = load_rcv1(TRAIN_FILE, TEST_FILE)

    X_test_eval = X_test[:TEST_EVAL_SIZE]
    y_test_eval = y_test[:TEST_EVAL_SIZE]

    print(f"Train shape : {X_train.shape}")
    print(f"Test shape  : {X_test.shape}")
    print(f"Batch size  : {BATCH_SIZE}")
    print(f"lambda      : {CAP_LAMBDA}")
    print(f"theta_cap   : {THETA_CAP}")

    L, max_sq_norm = estimate_component_L(X_train)
    print(f"max ||x_i||^2 = {max_sq_norm:.8f}")
    print(f"component L   = {L:.8f}")

    methods = ["Prox-SGD", "Prox-SVRG", "Prox-DASVRDA"]
    results = {}

    for method in methods:
        results[method] = run_dca(
            method=method,
            X_train=X_train,
            y_train=y_train,
            X_test_eval=X_test_eval,
            y_test_eval=y_test_eval,
            lam=CAP_LAMBDA,
            theta_cap=THETA_CAP,
            dca_outer=DCA_OUTER,
            batch_size=BATCH_SIZE,
            seed=SEED,
            sgd_lr=SGD_LR,
            svrg_lr=SVRG_LR,
            das_step_scale=DAS_STEP_SCALE,
            inner_stages=INNER_STAGES,
            sgd_inner_epochs=SGD_INNER_EPOCHS,
            L=L
        )

    print("\n" + "=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)

    summary_rows = []

    for method in methods:
        r = results[method]
        final_full_acc = accuracy(X_test, y_test, r["w"])
        final_obj = r["objective"][-1]
        final_sp = r["sparsity"][-1]
        final_grad = int(r["grad"][-1])
        final_time = float(r["time"][-1])

        print(
            f"{method:15s} | Obj={final_obj:.10f} | "
            f"Full Test Acc={final_full_acc:.4f} | "
            f"Sparsity={final_sp:.4f} | Grad={final_grad:,} | "
            f"Time={final_time:.3f}s"
        )

        summary_rows.append([
            method, final_obj, final_full_acc, final_sp,
            final_grad, final_time
        ])

    with open("rcv1_dca_final_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Method",
            "Final Nonconvex Objective",
            "Full Test Accuracy",
            "Sparsity",
            "Component Gradient Evaluations",
            "Optimization Time"
        ])
        writer.writerows(summary_rows)

    with open("rcv1_dca_trajectory.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Method",
            "DCA Iteration",
            "Nonconvex Objective",
            "Eval Test Accuracy",
            "Sparsity",
            "Component Gradient Evaluations",
            "Optimization Time"
        ])

        for method in methods:
            r = results[method]
            for i in range(len(r["objective"])):
                writer.writerow([
                    method,
                    i + 1,
                    r["objective"][i],
                    r["accuracy"][i],
                    r["sparsity"][i],
                    r["grad"][i],
                    r["time"][i]
                ])

    np.savez(
        "rcv1_dca_results.npz",
        cap_lambda=CAP_LAMBDA,
        theta_cap=THETA_CAP,
        sgd_objective=results["Prox-SGD"]["objective"],
        sgd_accuracy=results["Prox-SGD"]["accuracy"],
        sgd_sparsity=results["Prox-SGD"]["sparsity"],
        sgd_grad=results["Prox-SGD"]["grad"],
        sgd_time=results["Prox-SGD"]["time"],
        svrg_objective=results["Prox-SVRG"]["objective"],
        svrg_accuracy=results["Prox-SVRG"]["accuracy"],
        svrg_sparsity=results["Prox-SVRG"]["sparsity"],
        svrg_grad=results["Prox-SVRG"]["grad"],
        svrg_time=results["Prox-SVRG"]["time"],
        das_objective=results["Prox-DASVRDA"]["objective"],
        das_accuracy=results["Prox-DASVRDA"]["accuracy"],
        das_sparsity=results["Prox-DASVRDA"]["sparsity"],
        das_grad=results["Prox-DASVRDA"]["grad"],
        das_time=results["Prox-DASVRDA"]["time"]
    )

    # 1) Nonconvex objective vs component gradient evaluations
    plt.figure(figsize=(9, 6))
    for method in methods:
        r = results[method]
        plt.plot(r["grad"], r["objective"], marker="o", label=method)

    plt.xlabel("Number of Component Gradient Evaluations")
    plt.ylabel("Nonconvex Objective")
    plt.title("RCV1 capped-L1: DCA Solver Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("rcv1_dca_objective_vs_grad.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 2) Sparsity vs DCA iteration
    plt.figure(figsize=(9, 6))
    for method in methods:
        r = results[method]
        x = np.arange(1, len(r["sparsity"]) + 1)
        plt.plot(x, r["sparsity"], marker="o", label=method)

    plt.xlabel("DCA Outer Iteration")
    plt.ylabel("Sparsity (Fraction of Zero Weights)")
    plt.title("RCV1 capped-L1: Sparsity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("rcv1_dca_sparsity.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 3) Test accuracy vs DCA iteration
    plt.figure(figsize=(9, 6))
    for method in methods:
        r = results[method]
        x = np.arange(1, len(r["accuracy"]) + 1)
        plt.plot(x, r["accuracy"], marker="o", label=method)

    plt.xlabel("DCA Outer Iteration")
    plt.ylabel("Test Accuracy (first 10,000 test samples)")
    plt.title("RCV1 capped-L1: Test Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("rcv1_dca_accuracy.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
