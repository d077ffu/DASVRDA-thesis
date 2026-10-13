import time
import csv
import numpy as np
import matplotlib.pyplot as plt

from rcv1_dca_compare import (
    load_rcv1,
    nonconvex_objective,
    capped_h_subgradient,
    solve_subproblem_prox_sgd,
    solve_subproblem_prox_svrg,
    solve_subproblem_prox_dasvrda,
    estimate_component_L,
    accuracy,
)

TRAIN_FILE = "rcv1_train.binary"
TEST_FILE = "rcv1_test.binary"

SEED = 42
BATCH_SIZE = 8

CAP_LAMBDA = 1e-4
THETA_CAP = 1e-2

MAX_DCA_OUTER = 30

# Practical convergence:
# relative objective change < TOL for 3 consecutive DCA iterations
TOL = 1e-4
CONSECUTIVE_CHECKS = 3

# Same inner gradient-evaluation budget per DCA iteration
INNER_STAGES = 2
SGD_INNER_EPOCHS = 3 * INNER_STAGES

SGD_LR = 1.0
SVRG_LR = 10.0
DAS_STEP_SCALE = 20.0


def run_until_convergence(method, X_train, y_train, X_test, y_test, L):
    d = X_train.shape[1]
    w = np.zeros(d)

    prev_obj = nonconvex_objective(
        X_train, y_train, w, CAP_LAMBDA, THETA_CAP
    )

    stable_count = 0
    converged = False

    obj_hist = [prev_obj]
    time_hist = [0.0]

    total_grad = 0
    start_time = time.perf_counter()

    print()
    print("=" * 80)
    print(method)
    print("=" * 80)
    print(f"Initial objective = {prev_obj:.10f}")

    for t in range(MAX_DCA_OUTER):
        u = capped_h_subgradient(w, CAP_LAMBDA, THETA_CAP)

        if method == "DCA + Prox-SGD":
            w_new, used_grad = solve_subproblem_prox_sgd(
                X_train, y_train, w, u, CAP_LAMBDA,
                BATCH_SIZE, SGD_LR, SGD_INNER_EPOCHS, SEED + t
            )

        elif method == "DCA + Prox-SVRG":
            w_new, used_grad = solve_subproblem_prox_svrg(
                X_train, y_train, w, u, CAP_LAMBDA,
                BATCH_SIZE, SVRG_LR, INNER_STAGES, SEED + t
            )

        elif method == "DCA + Prox-DASVRDA":
            w_new, used_grad, _, _ = solve_subproblem_prox_dasvrda(
                X_train, y_train, w, u, CAP_LAMBDA,
                BATCH_SIZE, INNER_STAGES, DAS_STEP_SCALE, SEED + t, L
            )

        else:
            raise ValueError(method)

        w = w_new
        total_grad += used_grad

        current_obj = nonconvex_objective(
            X_train, y_train, w, CAP_LAMBDA, THETA_CAP
        )

        elapsed = time.perf_counter() - start_time

        relative_change = abs(current_obj - prev_obj) / max(abs(prev_obj), 1e-12)

        obj_hist.append(current_obj)
        time_hist.append(elapsed)

        print(
            f"DCA {t+1:02d} | "
            f"Obj={current_obj:.10f} | "
            f"RelChange={relative_change:.3e} | "
            f"Time={elapsed:.3f}s"
        )

        if relative_change < TOL:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= CONSECUTIVE_CHECKS:
            converged = True
            break

        prev_obj = current_obj

    convergence_time = time.perf_counter() - start_time

    final_accuracy = accuracy(X_test, y_test, w)

    return {
        "method": method,
        "converged": converged,
        "iterations": t + 1,
        "time": convergence_time,
        "accuracy": final_accuracy,
        "objective": current_obj,
        "grad_evals": total_grad,
        "obj_hist": np.array(obj_hist),
        "time_hist": np.array(time_hist)
    }


def main():
    print("Loading RCV1 ...")

    X_train, y_train, X_test, y_test = load_rcv1(TRAIN_FILE, TEST_FILE)

    print(f"Train shape : {X_train.shape}")
    print(f"Test shape  : {X_test.shape}")
    print(f"batch size  : {BATCH_SIZE}")
    print(f"lambda      : {CAP_LAMBDA}")
    print(f"theta       : {THETA_CAP}")
    print(f"tolerance   : {TOL}")
    print(f"stable checks: {CONSECUTIVE_CHECKS}")

    L, max_sq_norm = estimate_component_L(X_train)

    print(f"max ||x_i||^2 = {max_sq_norm:.8f}")
    print(f"L estimate     = {L:.8f}")

    methods = [
        "DCA + Prox-SGD",
        "DCA + Prox-SVRG",
        "DCA + Prox-DASVRDA"
    ]

    results = []

    for method in methods:
        results.append(
            run_until_convergence(
                method, X_train, y_train, X_test, y_test, L
            )
        )

    print()
    print("=" * 96)
    print("FINAL SUMMARY")
    print("=" * 96)

    for r in results:
        status = "Converged" if r["converged"] else "MaxIter"

        print(
            f"{r['method']:22s} | "
            f"{status:9s} | "
            f"Iter={r['iterations']:2d} | "
            f"Time={r['time']:.3f}s | "
            f"Accuracy={r['accuracy']:.4f} | "
            f"Obj={r['objective']:.8f}"
        )

    with open("rcv1_dca_time_accuracy_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Method",
            "Converged",
            "DCA Iterations",
            "Convergence Time [s]",
            "Final Test Accuracy",
            "Final Nonconvex Objective",
            "Component Gradient Evaluations"
        ])

        for r in results:
            writer.writerow([
                r["method"],
                r["converged"],
                r["iterations"],
                r["time"],
                r["accuracy"],
                r["objective"],
                r["grad_evals"]
            ])

    names = ["SGD", "SVRG", "DASVRDA"]

    # Convergence time
    plt.figure(figsize=(8, 6))
    plt.bar(names, [r["time"] for r in results])
    plt.ylabel("Time to Convergence [s]")
    plt.title("RCV1 Nonconvex DCA: Convergence Time")
    plt.tight_layout()
    plt.savefig("rcv1_dca_convergence_time.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Accuracy
    accs = [100.0 * r["accuracy"] for r in results]
    plt.figure(figsize=(8, 6))
    plt.bar(names, accs)
    plt.ylabel("Test Accuracy [%]")
    plt.title("RCV1 Nonconvex DCA: Test Accuracy")
    lower = max(0.0, min(accs) - 1.0)
    upper = min(100.0, max(accs) + 1.0)
    if upper > lower:
        plt.ylim(lower, upper)
    plt.tight_layout()
    plt.savefig("rcv1_dca_final_accuracy.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Optional convergence evidence
    plt.figure(figsize=(9, 6))
    for r, name in zip(results, names):
        plt.plot(r["time_hist"], r["obj_hist"], marker="o", label=name)
    plt.xlabel("Time [s]")
    plt.ylabel("Nonconvex Objective")
    plt.title("RCV1 Nonconvex DCA: Objective vs Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("rcv1_dca_objective_vs_time.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
