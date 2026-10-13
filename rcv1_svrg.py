import numpy as np
import matplotlib.pyplot as plt
import os
import time

from sklearn.datasets import load_svmlight_file
from scipy.special import expit


# =========================================================
# 1. RCV1データの読み込み
# =========================================================
def load_rcv1(train_file, test_file):

    print("RCV1データを読み込み中...")

    if not os.path.exists(train_file):
        raise FileNotFoundError(
            f"{train_file} が見つかりません。"
        )

    if not os.path.exists(test_file):
        raise FileNotFoundError(
            f"{test_file} が見つかりません。"
        )

    X_train, y_train = load_svmlight_file(
        train_file
    )

    X_test, y_test = load_svmlight_file(
        test_file,
        n_features=X_train.shape[1]
    )

    # Sparseのまま扱う
    X_train = X_train.tocsr()
    X_test = X_test.tocsr()

    return X_train, y_train, X_test, y_test


# =========================================================
# 2. 目的関数
# =========================================================
def compute_loss(X, w, y, lam):

    scores = X.dot(w)

    tmp = -y * scores

    logistic_loss = np.logaddexp(
        0,
        tmp
    )

    loss = np.mean(
        logistic_loss
    )

    regularization = (
        0.5
        * lam
        * np.sum(w ** 2)
    )

    return loss + regularization


# =========================================================
# 3. Accuracy
# =========================================================
def compute_accuracy(X, w, y):

    scores = X.dot(w)

    y_pred = np.where(
        scores >= 0,
        1,
        -1
    )

    return np.mean(
        y_pred == y
    )


# =========================================================
# 4. データ項だけのMini-batch勾配
# =========================================================
def mini_batch_data_gradient(
    X_batch,
    y_batch,
    w
):

    scores = X_batch.dot(w)

    coeff = (
        -y_batch
        * expit(
            -y_batch * scores
        )
    )

    grad = (
        X_batch.T.dot(coeff)
        / X_batch.shape[0]
    )

    grad = np.asarray(
        grad
    ).ravel()

    return grad


# =========================================================
# 5. 全勾配
# =========================================================
def full_gradient(
    X,
    y,
    w,
    lam
):

    scores = X.dot(w)

    coeff = (
        -y
        * expit(
            -y * scores
        )
    )

    grad = (
        X.T.dot(coeff)
        / X.shape[0]
    )

    grad = np.asarray(
        grad
    ).ravel()

    # L2正則化
    grad += lam * w

    return grad


# =========================================================
# 6. Mini-batch SVRG
# =========================================================
def mini_batch_svrg(
    X_train,
    y_train,
    X_test,
    y_test,
    lam=1e-4,
    learning_rate=0.1,
    batch_size=128,
    outer_iters=15,
    test_eval_size=10000,
    seed=42
):

    np.random.seed(seed)

    n, d = X_train.shape

    # ==========================================
    # 初期値
    # ==========================================
    w = np.zeros(d)

    # ==========================================
    # Test評価用データ
    # ==========================================
    test_eval_size = min(
        test_eval_size,
        X_test.shape[0]
    )

    X_test_eval = X_test[
        :test_eval_size
    ]

    y_test_eval = y_test[
        :test_eval_size
    ]

    # ==========================================
    # 履歴
    # ==========================================
    train_loss_hist = []
    test_acc_hist = []
    time_hist = []
    grad_count_hist = []

    optimization_time = 0.0
    grad_count = 0

    # ==========================================
    # 条件表示
    # ==========================================
    print()
    print("========================================")
    print("RCV1 Mini-batch SVRG")
    print("========================================")
    print(f"Train samples   : {n}")
    print(f"Features        : {d}")
    print(f"Batch size      : {batch_size}")
    print(f"Outer iterations: {outer_iters}")
    print(f"lambda          : {lam}")
    print(f"Learning rate   : {learning_rate}")
    print(f"Test eval size  : {test_eval_size}")
    print("========================================")
    print()

    # ==========================================
    # 初期状態
    # ==========================================
    initial_train_loss = compute_loss(
        X_train,
        w,
        y_train,
        lam
    )

    initial_test_acc = compute_accuracy(
        X_test_eval,
        w,
        y_test_eval
    )

    train_loss_hist.append(
        initial_train_loss
    )

    test_acc_hist.append(
        initial_test_acc
    )

    time_hist.append(0.0)

    grad_count_hist.append(0)

    print(
        f"Initial | "
        f"Train Loss: {initial_train_loss:.6f} | "
        f"Test Acc: {initial_test_acc:.4f}"
    )

    print()
    print("SVRG学習開始...")
    print()

    # =====================================================
    # Outer loop
    # =====================================================
    for outer in range(outer_iters):

        start_time = time.perf_counter()

        # ==========================================
        # ① 基準点 snapshot を固定
        # ==========================================
        snapshot = w.copy()

        # ==========================================
        # ② 基準点で全勾配
        # ==========================================
        full_grad = full_gradient(
            X_train,
            y_train,
            snapshot,
            lam
        )

        # 全データn件を使用
        grad_count += n

        # ==========================================
        # ③ 内側ループ用のデータ順
        # ==========================================
        indices = np.random.permutation(n)

        # ==========================================
        # ④ Mini-batch inner loop
        # ==========================================
        for start in range(
            0,
            n,
            batch_size
        ):

            end = min(
                start + batch_size,
                n
            )

            batch_indices = indices[
                start:end
            ]

            X_batch = X_train[
                batch_indices
            ]

            y_batch = y_train[
                batch_indices
            ]

            current_batch_size = len(
                batch_indices
            )

            # --------------------------------------
            # 現在位置 w での勾配
            # --------------------------------------
            grad_current = mini_batch_data_gradient(
                X_batch,
                y_batch,
                w
            )

            # --------------------------------------
            # 基準点 snapshot での
            # 同じMini-batchの勾配
            # --------------------------------------
            grad_snapshot = mini_batch_data_gradient(
                X_batch,
                y_batch,
                snapshot
            )

            # currentとsnapshotの2回
            grad_count += (
                2 * current_batch_size
            )

            # ======================================
            # ⑤ SVRGの分散削減勾配
            # ======================================
            #
            # v =
            # ∇F(snapshot)
            # + ∇f_B(w)
            # - ∇f_B(snapshot)
            #
            # L2も補正
            # ======================================
            v = (
                full_grad
                + grad_current
                - grad_snapshot
                + lam * (
                    w - snapshot
                )
            )

            # ======================================
            # ⑥ 重み更新
            # ======================================
            w = (
                w
                - learning_rate * v
            )

        # ==========================================
        # 最適化時間
        # ==========================================
        end_time = time.perf_counter()

        optimization_time += (
            end_time
            - start_time
        )

        # ==========================================
        # 評価
        # ==========================================
        train_loss = compute_loss(
            X_train,
            w,
            y_train,
            lam
        )

        test_acc = compute_accuracy(
            X_test_eval,
            w,
            y_test_eval
        )

        train_loss_hist.append(
            train_loss
        )

        test_acc_hist.append(
            test_acc
        )

        time_hist.append(
            optimization_time
        )

        grad_count_hist.append(
            grad_count
        )

        print(
            f"Outer {outer + 1:2d}/{outer_iters} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Test Acc: {test_acc:.4f} | "
            f"Time: {optimization_time:.3f}s | "
            f"Grad Eval: {grad_count}"
        )

    # =====================================================
    # 最終評価
    # =====================================================
    final_train_loss = compute_loss(
        X_train,
        w,
        y_train,
        lam
    )

    print()
    print("全Testデータで最終Accuracyを計算中...")

    final_test_acc_full = compute_accuracy(
        X_test,
        w,
        y_test
    )

    print()
    print("========================================")
    print("SVRG Final Result")
    print("========================================")

    print(
        f"Final Train Loss    : "
        f"{final_train_loss:.6f}"
    )

    print(
        f"Final Test Accuracy : "
        f"{final_test_acc_full:.4f}"
    )

    print(
        f"Optimization Time   : "
        f"{optimization_time:.3f} sec"
    )

    print(
        f"Gradient Evals      : "
        f"{grad_count}"
    )

    print("========================================")

    return (
        w,
        train_loss_hist,
        test_acc_hist,
        time_hist,
        grad_count_hist,
        final_test_acc_full
    )


# =========================================================
# 7. メイン
# =========================================================
if __name__ == "__main__":

    train_file = "rcv1_train.binary"
    test_file = "rcv1_test.binary"

    # ==========================================
    # データ読み込み
    # ==========================================
    (
        X_train,
        y_train,
        X_test,
        y_test
    ) = load_rcv1(
        train_file,
        test_file
    )

    print()
    print("Dataset information")
    print("----------------------------------------")
    print(
        "Train shape:",
        X_train.shape
    )
    print(
        "Test shape :",
        X_test.shape
    )
    print(
        "Train type :",
        type(X_train)
    )
    print("----------------------------------------")

    # ==========================================
    # ハイパーパラメータ
    # ==========================================
    lambda_param = 1e-4

    learning_rate = 0.1

    batch_size = 128

    # 15 outer
    #
    # 1 outer:
    # Full gradient   = n
    # Inner current   = n
    # Inner snapshot  = n
    #
    # 合計 = 3n
    #
    # 15 × 3 × 20242
    # = 910890
    outer_iterations = 15

    test_eval_size = 10000

    # ==========================================
    # SVRG実行
    # ==========================================
    (
        w_opt,
        train_loss_history,
        test_acc_history,
        time_history,
        grad_count_history,
        final_test_acc
    ) = mini_batch_svrg(
        X_train,
        y_train,
        X_test,
        y_test,
        lam=lambda_param,
        learning_rate=learning_rate,
        batch_size=batch_size,
        outer_iters=outer_iterations,
        test_eval_size=test_eval_size,
        seed=42
    )

    # =====================================================
    # 結果保存
    # =====================================================
    np.savez(
        "rcv1_svrg_results.npz",

        train_loss=np.array(
            train_loss_history
        ),

        test_accuracy=np.array(
            test_acc_history
        ),

        time=np.array(
            time_history
        ),

        grad_count=np.array(
            grad_count_history
        )
    )

    print()
    print(
        "結果を rcv1_svrg_results.npz に保存しました。"
    )

    # =====================================================
    # グラフ1
    # Train Loss vs Time
    # =====================================================
    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        time_history,
        train_loss_history,
        marker="o"
    )

    plt.xlabel(
        "Optimization Time [s]"
    )

    plt.ylabel(
        "Train Loss"
    )

    plt.title(
        "RCV1 SVRG: Train Loss vs Time"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "rcv1_svrg_time_loss.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # =====================================================
    # グラフ2
    # Train Loss vs Gradient Evaluations
    # =====================================================
    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        grad_count_history,
        train_loss_history,
        marker="o"
    )

    plt.xlabel(
        "Number of Component Gradient Evaluations"
    )

    plt.ylabel(
        "Train Loss"
    )

    plt.title(
        "RCV1 SVRG: Train Loss vs Gradient Evaluations"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "rcv1_svrg_gradient_loss.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # =====================================================
    # グラフ3
    # Test Accuracy vs Time
    # =====================================================
    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        time_history,
        test_acc_history,
        marker="o"
    )

    plt.xlabel(
        "Optimization Time [s]"
    )

    plt.ylabel(
        "Test Accuracy"
    )

    plt.title(
        "RCV1 SVRG: Test Accuracy vs Time"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "rcv1_svrg_time_accuracy.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()