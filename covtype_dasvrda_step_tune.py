import csv
import time
import numpy as np
import matplotlib.pyplot as plt

from covtype_dca_batch_compare import (
    load_covtype,
    estimate_component_L,
    nonconvex_objective,
    capped_h_subgradient,
    prox_dasvrda,
)

# ============================================================
# DASVRDA step-scale tuning for each mini-batch size
# Tune ONLY by training nonconvex objective.
#
# Same oracle budget for every run:
#   DCA outer = 4
#   DASVRDA inner stage = 1
#   => 3n component-gradient evaluations per DCA outer
# ============================================================

SEED = 42

BATCH_SIZES = [128, 512, 2048]

STEP_SCALES = [
    10.0,
    20.0,
    40.0,
    80.0,
    160.0,
    320.0,
    640.0,
]

CAP_LAMBDA = 1e-4
THETA_CAP = 1e-2

TUNE_DCA_OUTER = 4
INNER_STAGES = 1


def run_one_setting(
    X_train,
    y_train,
    batch_size,
    step_scale,
    L,
):
    w = np.zeros(
        X_train.shape[1],
        dtype=np.float32
    )

    initial_obj = nonconvex_objective(
        X_train,
        y_train,
        w,
        CAP_LAMBDA,
        THETA_CAP
    )

    total_grad = 0
    obj_hist = [initial_obj]

    t0 = time.perf_counter()

    for outer in range(TUNE_DCA_OUTER):

        u = capped_h_subgradient(
            w,
            CAP_LAMBDA,
            THETA_CAP
        )

        w, used = prox_dasvrda(
            X_train,
            y_train,
            w,
            u,
            CAP_LAMBDA,
            batch_size,
            INNER_STAGES,
            step_scale,
            SEED + outer,
            L
        )

        total_grad += used

        obj = nonconvex_objective(
            X_train,
            y_train,
            w,
            CAP_LAMBDA,
            THETA_CAP
        )

        obj_hist.append(obj)

        if not np.isfinite(obj):
            break

    elapsed = time.perf_counter() - t0

    final_obj = obj_hist[-1]

    return {
        "batch_size": batch_size,
        "step_scale": step_scale,
        "initial_objective": initial_obj,
        "final_objective": final_obj,
        "objective_drop": initial_obj - final_obj,
        "time": elapsed,
        "grad_evals": total_grad,
        "objective_history": np.array(obj_hist),
    }


def main():

    X_train, y_train, X_test, y_test = load_covtype()

    print()
    print("Train:", X_train.shape)
    print("Test :", X_test.shape)

    L, max_sq_norm = estimate_component_L(
        X_train
    )

    print(f"max ||x_i||^2 = {max_sq_norm:.6f}")
    print(f"L estimate     = {L:.6f}")
    print()

    results = []

    for batch_size in BATCH_SIZES:

        print()
        print("#" * 88)
        print(f"BATCH SIZE = {batch_size}")
        print("#" * 88)

        for step_scale in STEP_SCALES:

            r = run_one_setting(
                X_train,
                y_train,
                batch_size,
                step_scale,
                L
            )

            results.append(r)

            print(
                f"scale={step_scale:7.1f} | "
                f"Final Obj={r['final_objective']:.8f} | "
                f"Drop={r['objective_drop']:.8f} | "
                f"Time={r['time']:.2f}s"
            )

    # ========================================================
    # Best setting for each batch
    # ========================================================
    print()
    print("=" * 92)
    print("BEST STEP SCALE FOR EACH BATCH")
    print("=" * 92)

    best_by_batch = {}

    for batch_size in BATCH_SIZES:

        rr = [
            r for r in results
            if r["batch_size"] == batch_size
            and np.isfinite(r["final_objective"])
        ]

        best = min(
            rr,
            key=lambda x: x["final_objective"]
        )

        best_by_batch[batch_size] = best

        print(
            f"batch={batch_size:4d} | "
            f"best scale={best['step_scale']:7.1f} | "
            f"Final Obj={best['final_objective']:.8f} | "
            f"Drop={best['objective_drop']:.8f} | "
            f"Time={best['time']:.2f}s"
        )

    # ========================================================
    # CSV
    # ========================================================
    with open(
        "covtype_dasvrda_step_tuning.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Batch Size",
            "Step Scale",
            "Initial Objective",
            "Final Objective",
            "Objective Drop",
            "Time [s]",
            "Component Gradient Evaluations"
        ])

        for r in results:
            writer.writerow([
                r["batch_size"],
                r["step_scale"],
                r["initial_objective"],
                r["final_objective"],
                r["objective_drop"],
                r["time"],
                r["grad_evals"]
            ])

    # ========================================================
    # Plot: final objective vs step scale
    # ========================================================
    plt.figure(figsize=(9, 6))

    for batch_size in BATCH_SIZES:

        rr = [
            r for r in results
            if r["batch_size"] == batch_size
        ]

        plt.plot(
            [r["step_scale"] for r in rr],
            [r["final_objective"] for r in rr],
            marker="o",
            label=f"batch={batch_size}"
        )

    plt.xscale("log", base=2)
    plt.xlabel("DASVRDA Step Scale")
    plt.ylabel("Final Nonconvex Objective")
    plt.title("Covertype: DASVRDA Step-Scale Tuning")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "covtype_dasvrda_step_tuning.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # ========================================================
    # Plot: objective trajectory for BEST scale of each batch
    # ========================================================
    plt.figure(figsize=(9, 6))

    for batch_size in BATCH_SIZES:

        best = best_by_batch[batch_size]

        x = np.arange(
            len(best["objective_history"])
        )

        plt.plot(
            x,
            best["objective_history"],
            marker="o",
            label=(
                f"batch={batch_size}, "
                f"scale={best['step_scale']:.0f}"
            )
        )

    plt.xlabel("DCA Outer Iteration")
    plt.ylabel("Nonconvex Objective")
    plt.title("Covertype: Tuned DASVRDA Objective")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "covtype_dasvrda_best_trajectory.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":
    main()
