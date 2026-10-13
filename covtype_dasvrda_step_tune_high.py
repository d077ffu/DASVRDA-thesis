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

SEED = 42

BATCH_SIZES = [128, 512, 2048]

# Previous search ended at 640 and 640 was best for all batches,
# so extend the search upward.
STEP_SCALES = [
    640.0,
    1280.0,
    2560.0,
    5120.0,
    10240.0,
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

    obj_hist = [initial_obj]
    total_grad = 0
    diverged = False

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

        # Safe stop if the run clearly becomes unstable.
        if (
            not np.isfinite(obj)
            or obj > initial_obj * 2.0
        ):
            diverged = True
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
        "diverged": diverged,
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
        print("#" * 90)
        print(f"BATCH SIZE = {batch_size}")
        print("#" * 90)

        for step_scale in STEP_SCALES:

            r = run_one_setting(
                X_train,
                y_train,
                batch_size,
                step_scale,
                L
            )

            results.append(r)

            status = "DIVERGED" if r["diverged"] else "OK"

            print(
                f"scale={step_scale:8.1f} | "
                f"Final Obj={r['final_objective']:.8f} | "
                f"Drop={r['objective_drop']:.8f} | "
                f"Time={r['time']:.2f}s | "
                f"{status}"
            )

    print()
    print("=" * 96)
    print("BEST STEP SCALE FOR EACH BATCH")
    print("=" * 96)

    best_by_batch = {}

    for batch_size in BATCH_SIZES:

        rr = [
            r for r in results
            if r["batch_size"] == batch_size
            and np.isfinite(r["final_objective"])
            and not r["diverged"]
        ]

        best = min(
            rr,
            key=lambda x: x["final_objective"]
        )

        best_by_batch[batch_size] = best

        print(
            f"batch={batch_size:4d} | "
            f"best scale={best['step_scale']:8.1f} | "
            f"Final Obj={best['final_objective']:.8f} | "
            f"Drop={best['objective_drop']:.8f} | "
            f"Time={best['time']:.2f}s"
        )

    with open(
        "covtype_dasvrda_step_tuning_high.csv",
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
            "Component Gradient Evaluations",
            "Diverged"
        ])

        for r in results:
            writer.writerow([
                r["batch_size"],
                r["step_scale"],
                r["initial_objective"],
                r["final_objective"],
                r["objective_drop"],
                r["time"],
                r["grad_evals"],
                r["diverged"]
            ])

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
    plt.title("Covertype: DASVRDA Extended Step-Scale Tuning")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "covtype_dasvrda_step_tuning_high.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

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
    plt.title("Covertype: Best DASVRDA Trajectory")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "covtype_dasvrda_best_trajectory_high.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":
    main()
