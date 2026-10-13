import csv
import time
import numpy as np
import matplotlib.pyplot as plt

from covtype_dca_batch_compare import (
    load_covtype,
    estimate_component_L,
    nonconvex_objective,
    capped_h_subgradient,
    prox_sgd,
    prox_svrg,
    prox_dasvrda,
    accuracy,
)

# ============================================================
# FINAL EXPERIMENT
# Covertype + capped-L1 + DCA
#
# Question:
# When mini-batch size becomes large, how does DASVRDA compare
# with SGD and SVRG in time to reach the SAME objective level?
# ============================================================

SEED = 42

BATCH_SIZES = [128, 512, 2048]

CAP_LAMBDA = 1e-4
THETA_CAP = 1e-2

# Common optimization target.
# All methods are judged by the same nonconvex-objective threshold.
TARGET_OBJECTIVE = 0.56

MAX_DCA_OUTER = 80

# Same component-gradient budget per DCA outer iteration:
# SGD: 3 epochs = 3n
# SVRG: 1 stage = n + 2n = 3n
# DASVRDA: 1 stage = n + 2n = 3n
SGD_INNER_EPOCHS = 3
INNER_STAGES = 1

# ------------------------------------------------------------
# Small TRAINING-OBJECTIVE-only tuning for SGD and SVRG.
# This is done before the final timing run.
# Tuning time is NOT included in final runtime.
# ------------------------------------------------------------
TUNE_DCA_OUTER = 4

SGD_LR_SCALES = [1.0, 4.0, 16.0, 64.0, 256.0]
SVRG_LR_SCALES = [4.0, 16.0, 64.0, 256.0, 1024.0]

# Best DASVRDA value among the scales already tested.
# It was best for batch 128, 512, 2048.
DAS_STEP_SCALE = 10240.0


def run_fixed_outer(
    method,
    X_train,
    y_train,
    batch_size,
    L,
    parameter,
    outer_iters,
):
    """Used only for hyperparameter tuning."""
    w = np.zeros(X_train.shape[1], dtype=np.float32)

    initial_obj = nonconvex_objective(
        X_train, y_train, w,
        CAP_LAMBDA, THETA_CAP
    )

    current_obj = initial_obj

    for outer in range(outer_iters):
        u = capped_h_subgradient(
            w, CAP_LAMBDA, THETA_CAP
        )

        if method == "SGD":
            lr = parameter / L

            w, _ = prox_sgd(
                X_train, y_train,
                w, u, CAP_LAMBDA,
                batch_size, lr,
                SGD_INNER_EPOCHS,
                SEED + outer
            )

        elif method == "SVRG":
            lr = parameter / L

            w, _ = prox_svrg(
                X_train, y_train,
                w, u, CAP_LAMBDA,
                batch_size, lr,
                INNER_STAGES,
                SEED + outer
            )

        elif method == "DASVRDA":
            w, _ = prox_dasvrda(
                X_train, y_train,
                w, u, CAP_LAMBDA,
                batch_size,
                INNER_STAGES,
                parameter,
                SEED + outer,
                L
            )

        current_obj = nonconvex_objective(
            X_train, y_train, w,
            CAP_LAMBDA, THETA_CAP
        )

        # Bad / unstable trial
        if (
            not np.isfinite(current_obj)
            or current_obj > initial_obj * 2.0
        ):
            return np.inf

    return current_obj


def tune_method(
    method,
    candidates,
    X_train,
    y_train,
    batch_size,
    L,
):
    print()
    print(f"TUNING {method} | batch={batch_size}")

    best_param = None
    best_obj = np.inf

    for p in candidates:
        obj = run_fixed_outer(
            method,
            X_train,
            y_train,
            batch_size,
            L,
            p,
            TUNE_DCA_OUTER
        )

        print(
            f"  parameter={p:10.3f} | "
            f"objective={obj:.8f}"
        )

        if obj < best_obj:
            best_obj = obj
            best_param = p

    print(
        f"  BEST -> parameter={best_param} | "
        f"objective={best_obj:.8f}"
    )

    return best_param, best_obj


def run_to_target(
    method,
    X_train,
    y_train,
    X_test,
    y_test,
    batch_size,
    L,
    parameter,
):
    w = np.zeros(X_train.shape[1], dtype=np.float32)

    initial_obj = nonconvex_objective(
        X_train, y_train, w,
        CAP_LAMBDA, THETA_CAP
    )

    total_grad = 0
    reached = False

    obj_hist = [initial_obj]
    time_hist = [0.0]

    t0 = time.perf_counter()

    for outer in range(MAX_DCA_OUTER):
        u = capped_h_subgradient(
            w, CAP_LAMBDA, THETA_CAP
        )

        if method == "SGD":
            lr = parameter / L

            w, used = prox_sgd(
                X_train, y_train,
                w, u, CAP_LAMBDA,
                batch_size, lr,
                SGD_INNER_EPOCHS,
                SEED + outer
            )

        elif method == "SVRG":
            lr = parameter / L

            w, used = prox_svrg(
                X_train, y_train,
                w, u, CAP_LAMBDA,
                batch_size, lr,
                INNER_STAGES,
                SEED + outer
            )

        elif method == "DASVRDA":
            w, used = prox_dasvrda(
                X_train, y_train,
                w, u, CAP_LAMBDA,
                batch_size,
                INNER_STAGES,
                parameter,
                SEED + outer,
                L
            )

        total_grad += used

        obj = nonconvex_objective(
            X_train, y_train, w,
            CAP_LAMBDA, THETA_CAP
        )

        elapsed = time.perf_counter() - t0

        obj_hist.append(obj)
        time_hist.append(elapsed)

        print(
            f"{method:8s} | "
            f"batch={batch_size:4d} | "
            f"DCA={outer+1:02d} | "
            f"Obj={obj:.8f} | "
            f"Time={elapsed:.3f}s"
        )

        if not np.isfinite(obj):
            break

        if obj <= TARGET_OBJECTIVE:
            reached = True
            break

    total_time = time.perf_counter() - t0

    # Evaluate test accuracy only after stopping.
    final_acc = accuracy(
        X_test, y_test, w
    )

    return {
        "method": method,
        "batch_size": batch_size,
        "parameter": parameter,
        "reached": reached,
        "outer_iters": outer + 1,
        "time": total_time,
        "accuracy": final_acc,
        "objective": obj,
        "grad_evals": total_grad,
        "obj_hist": np.array(obj_hist),
        "time_hist": np.array(time_hist),
    }


def main():

    X_train, y_train, X_test, y_test = load_covtype()

    print()
    print("=" * 95)
    print("FINAL EXPERIMENT: Covertype + Nonconvex DCA")
    print("=" * 95)
    print("Train shape       :", X_train.shape)
    print("Test shape        :", X_test.shape)
    print("Batch sizes       :", BATCH_SIZES)
    print("Target objective  :", TARGET_OBJECTIVE)
    print("DAS step scale    :", DAS_STEP_SCALE)

    L, max_sq_norm = estimate_component_L(X_train)

    print("max ||x_i||^2     :", max_sq_norm)
    print("L estimate        :", L)
    print()

    # ========================================================
    # 1. Fair preliminary tuning
    # ========================================================
    best_parameters = {}

    for batch_size in BATCH_SIZES:

        sgd_scale, _ = tune_method(
            "SGD",
            SGD_LR_SCALES,
            X_train,
            y_train,
            batch_size,
            L
        )

        svrg_scale, _ = tune_method(
            "SVRG",
            SVRG_LR_SCALES,
            X_train,
            y_train,
            batch_size,
            L
        )

        best_parameters[("SGD", batch_size)] = sgd_scale
        best_parameters[("SVRG", batch_size)] = svrg_scale
        best_parameters[("DASVRDA", batch_size)] = DAS_STEP_SCALE

    print()
    print("=" * 95)
    print("SELECTED PARAMETERS")
    print("=" * 95)

    for batch_size in BATCH_SIZES:
        print(
            f"batch={batch_size:4d} | "
            f"SGD scale={best_parameters[('SGD', batch_size)]} | "
            f"SVRG scale={best_parameters[('SVRG', batch_size)]} | "
            f"DAS scale={best_parameters[('DASVRDA', batch_size)]}"
        )

    # ========================================================
    # 2. Final time-to-target experiment
    # ========================================================
    methods = ["SGD", "SVRG", "DASVRDA"]
    results = []

    for batch_size in BATCH_SIZES:

        print()
        print("#" * 95)
        print(f"FINAL RUN | BATCH SIZE = {batch_size}")
        print("#" * 95)

        for method in methods:

            parameter = best_parameters[
                (method, batch_size)
            ]

            r = run_to_target(
                method,
                X_train,
                y_train,
                X_test,
                y_test,
                batch_size,
                L,
                parameter
            )

            results.append(r)

    # ========================================================
    # 3. Summary
    # ========================================================
    print()
    print("=" * 115)
    print("FINAL SUMMARY")
    print("=" * 115)

    for r in results:

        status = (
            "REACHED"
            if r["reached"]
            else "NOT REACHED"
        )

        print(
            f"{r['method']:8s} | "
            f"batch={r['batch_size']:4d} | "
            f"{status:11s} | "
            f"Outer={r['outer_iters']:2d} | "
            f"Time={r['time']:.3f}s | "
            f"Acc={100*r['accuracy']:.3f}% | "
            f"Obj={r['objective']:.8f} | "
            f"Param={r['parameter']}"
        )

    # ========================================================
    # 4. Save CSV
    # ========================================================
    with open(
        "covtype_dca_main_results.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Method",
            "Batch Size",
            "Parameter",
            "Target Objective",
            "Reached Target",
            "DCA Outer Iterations",
            "Time to Stop [s]",
            "Test Accuracy",
            "Final Nonconvex Objective",
            "Component Gradient Evaluations"
        ])

        for r in results:
            writer.writerow([
                r["method"],
                r["batch_size"],
                r["parameter"],
                TARGET_OBJECTIVE,
                r["reached"],
                r["outer_iters"],
                r["time"],
                r["accuracy"],
                r["objective"],
                r["grad_evals"]
            ])

    # ========================================================
    # 5. Plot: time to common target
    # ========================================================
    plt.figure(figsize=(9, 6))

    for method in methods:

        rr = [
            r for r in results
            if r["method"] == method
        ]

        x = [
            r["batch_size"]
            for r in rr
        ]

        y = [
            r["time"] if r["reached"] else np.nan
            for r in rr
        ]

        plt.plot(
            x,
            y,
            marker="o",
            label=method
        )

    plt.xscale("log", base=2)

    plt.xlabel("Mini-batch Size")
    plt.ylabel("Time to Target Objective [s]")

    plt.title(
        f"Covertype + DCA: Time to Objective <= {TARGET_OBJECTIVE}"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "covtype_dca_main_time_to_target.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # ========================================================
    # 6. Plot: test accuracy at stopping point
    # ========================================================
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

    plt.title(
        "Covertype + DCA: Test Accuracy at Stopping Point"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "covtype_dca_main_accuracy.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # ========================================================
    # 7. Optional: all objective-time trajectories
    # ========================================================
    for batch_size in BATCH_SIZES:

        plt.figure(figsize=(9, 6))

        for method in methods:

            r = next(
                x for x in results
                if x["method"] == method
                and x["batch_size"] == batch_size
            )

            plt.plot(
                r["time_hist"],
                r["obj_hist"],
                marker="o",
                label=method
            )

        plt.axhline(
            TARGET_OBJECTIVE,
            linestyle="--",
            label="Target"
        )

        plt.xlabel("Elapsed Time [s]")
        plt.ylabel("Nonconvex Objective")

        plt.title(
            f"Covertype + DCA: Objective vs Time (batch={batch_size})"
        )

        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            f"covtype_dca_objective_time_batch{batch_size}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


if __name__ == "__main__":
    main()
