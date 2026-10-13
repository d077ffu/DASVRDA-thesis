import numpy as np
import matplotlib.pyplot as plt
import os
import time

from sklearn.datasets import load_svmlight_file
from scipy.special import expit


# =========================================================
# 1. RCV1データ読み込み
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

    # Sparseのまま使用
    X_train = X_train.tocsr()
    X_test = X_test.tocsr()

    return X_train, y_train, X_test, y_test


# =========================================================
# 2. Loss
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
# 4. Mini-batchのデータ項勾配
#    ※L2正則化はここには入れない
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
# 6. Lipschitz定数の上界を計算
# =========================================================
def estimate_L(X, lam):

    # 各データ ||x_i||^2
    row_norm_sq = np.asarray(
        X.multiply(X).sum(axis=1)
    ).ravel()

    max_norm_sq = np.max(
        row_norm_sq
    )

    # Logistic loss:
    # L_i <= 1/4 * ||x_i||^2 + lambda
    L = (
        0.25 * max_norm_sq
        + lam
    )

    return L, max_norm_sq


# =========================================================
# 7. Mini-batch DASVRDA
# =========================================================
def mini_batch_dasvrda(
    X_train,
    y_train,
    X_test,
    y_test,
    lam=1e-4,
    batch_size=128,
    outer_iters=15,
    test_eval_size=10000,
    seed=42,
    step_scale=1.0
):

    np.random.seed(seed)

    n, d = X_train.shape

    # -----------------------------------------
    # Mini-batch数
    # -----------------------------------------
    m_inner = int(
        np.ceil(
            n / batch_size
        )
    )

    # -----------------------------------------
    # Lipschitz定数
    # -----------------------------------------
    L, max_norm_sq = estimate_L(
        X_train,
        lam
    )

    # -----------------------------------------
    # DASVRDAパラメータ
    # -----------------------------------------
    gamma = (
        3.0
        + np.sqrt(
            9.0
            + 8.0 / (m_inner + 1)
        )
    ) / 2.0

    base_step_size = (
        1.0
        /
        (
            (
                1.0
                + gamma * (m_inner + 1)
            )
            * L
        )
    )

    # 後で調整できるように倍率を用意
    step_size = (
        step_scale
        * base_step_size
    )

    # =========================================
    # 初期値
    # =========================================
    w = np.zeros(d)

    # 現在の基準点
    wold = np.zeros(d)

    # 1つ前の基準点
    wold_ = np.zeros(d)

    # 前outerの補助変数
    z = np.zeros(d)

    # =========================================
    # Test評価用
    # =========================================
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

    # =========================================
    # 履歴
    # =========================================
    train_loss_hist = []
    test_acc_hist = []
    time_hist = []
    grad_count_hist = []

    optimization_time = 0.0
    grad_count = 0

    # =========================================
    # 条件表示
    # =========================================
    print()
    print("========================================")
    print("RCV1 Mini-batch DASVRDA")
    print("========================================")
    print(f"Train samples    : {n}")
    print(f"Features         : {d}")
    print(f"Batch size       : {batch_size}")
    print(f"Inner batches    : {m_inner}")
    print(f"Outer iterations : {outer_iters}")
    print(f"lambda           : {lam}")
    print(f"max ||x_i||^2    : {max_norm_sq:.6f}")
    print(f"Estimated L      : {L:.6f}")
    print(f"gamma            : {gamma:.6f}")
    print(f"Base step size   : {base_step_size:.8f}")
    print(f"Step scale       : {step_scale}")
    print(f"Actual step size : {step_size:.8f}")
    print(f"Test eval size   : {test_eval_size}")
    print("========================================")
    print()

    # =========================================
    # 初期状態
    # =========================================
    initial_loss = compute_loss(
        X_train,
        wold,
        y_train,
        lam
    )

    initial_acc = compute_accuracy(
        X_test_eval,
        wold,
        y_test_eval
    )

    train_loss_hist.append(
        initial_loss
    )

    test_acc_hist.append(
        initial_acc
    )

    time_hist.append(0.0)
    grad_count_hist.append(0)

    print(
        f"Initial | "
        f"Train Loss: {initial_loss:.6f} | "
        f"Test Acc: {initial_acc:.4f}"
    )

    print()
    print("DASVRDA学習開始...")
    print()

    # =====================================================
    # Outer loop
    # =====================================================
    for k in range(outer_iters):

        start_time = time.perf_counter()

        # =================================================
        # ① 外側加速
        #    ここが「二重加速」の外側部分
        # =================================================
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

        weight1 = (
            (thetas_old - 1.0)
            / thetas
        )

        weight2 = (
            thetas_old
            / thetas
        )

        w = (
            wold
            + weight1
            * (wold - wold_)
            + weight2
            * (z - wold)
        )

        # =================================================
        # ② 基準点 wold で全勾配
        # =================================================
        gold = full_gradient(
            X_train,
            y_train,
            wold,
            lam
        )

        # 全データn個
        grad_count += n

        # =================================================
        # ③ 内側ループ初期化
        # =================================================
        avg = np.zeros(d)

        z_inner = w.copy()

        # Dual Averagingの固定基準点
        z0 = w.copy()

        # 1outerで全データを1回通す
        indices = np.random.permutation(n)

        # =================================================
        # Inner loop
        # =================================================
        inner_id = 0

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

            # ---------------------------------------------
            # ④ theta
            # ---------------------------------------------
            theta = (
                2.0
                / (inner_id + 2.0)
            )

            theta_old = (
                2.0
                / (inner_id + 1.0)
            )

            # ---------------------------------------------
            # ⑤ y_k
            #
            # 内側加速の評価点
            # ---------------------------------------------
            y_point = (
                (1.0 - theta) * w
                + theta * z_inner
            )

            # ---------------------------------------------
            # ⑥ 現在位置 y_k のMini-batch勾配
            # ---------------------------------------------
            grad_current = mini_batch_data_gradient(
                X_batch,
                y_batch,
                y_point
            )

            # ---------------------------------------------
            # ⑦ 基準点 wold の同じMini-batch勾配
            # ---------------------------------------------
            grad_snapshot = mini_batch_data_gradient(
                X_batch,
                y_batch,
                wold
            )

            # current + snapshot
            grad_count += (
                2 * current_batch_size
            )

            # ---------------------------------------------
            # ⑧ 分散削減勾配
            #
            # g_k =
            # ∇F(wold)
            # + ∇f_B(y_k)
            # - ∇f_B(wold)
            # ---------------------------------------------
            g = (
                gold
                + grad_current
                - grad_snapshot
                + lam * (
                    y_point - wold
                )
            )

            # ---------------------------------------------
            # ⑨ 平均勾配
            # ---------------------------------------------
            avg = (
                (1.0 - theta) * avg
                + theta * g
            )

            # ---------------------------------------------
            # ⑩ Dual Averaging
            #
            # ここがDual Averagingの中心
            # ---------------------------------------------
            z_inner = (
                z0
                - step_size
                * avg
                / (
                    theta
                    * theta_old
                )
            )

            # ---------------------------------------------
            # ⑪ w更新
            #
            # ここが内側加速に関わる部分
            # ---------------------------------------------
            w = (
                (1.0 - theta) * w
                + theta * z_inner
            )

            inner_id += 1

        # =================================================
        # Inner loop終了
        # =================================================

        # 基準点更新
        wold_ = wold.copy()
        wold = w.copy()

        # 次のouterで使うz
        z = z_inner.copy()

        end_time = time.perf_counter()

        optimization_time += (
            end_time
            - start_time
        )

        # =================================================
        # 評価
        # =================================================
        train_loss = compute_loss(
            X_train,
            wold,
            y_train,
            lam
        )

        test_acc = compute_accuracy(
            X_test_eval,
            wold,
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
            f"Outer {k + 1:2d}/{outer_iters} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Test Acc: {test_acc:.4f} | "
            f"Time: {optimization_time:.3f}s | "
            f"Grad Eval: {grad_count}"
        )

        # 発散チェック
        if not np.isfinite(train_loss):

            print()
            print(
                "Lossがnanまたはinfになりました。"
            )

            print(
                "step_scaleを小さくしてください。"
            )

            break

    # =====================================================
    # 最終評価
    # =====================================================
    final_train_loss = compute_loss(
        X_train,
        wold,
        y_train,
        lam
    )

    print()
    print(
        "全Testデータで最終Accuracyを計算中..."
    )

    final_test_acc = compute_accuracy(
        X_test,
        wold,
        y_test
    )

    print()
    print("========================================")
    print("DASVRDA Final Result")
    print("========================================")

    print(
        f"Final Train Loss    : "
        f"{final_train_loss:.6f}"
    )

    print(
        f"Final Test Accuracy : "
        f"{final_test_acc:.4f}"
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
        wold,
        train_loss_hist,
        test_acc_hist,
        time_hist,
        grad_count_hist,
        final_test_acc
    )


# =========================================================
# 8. Main
# =========================================================
if __name__ == "__main__":

    train_file = "rcv1_train.binary"
    test_file = "rcv1_test.binary"

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
    print("Train shape:", X_train.shape)
    print("Test shape :", X_test.shape)
    print("Train type :", type(X_train))
    print("----------------------------------------")

    # =====================================================
    # 共通条件
    # =====================================================
    lambda_param = 1e-4

    batch_size = 128

    # 1 outerあたり
    #
    # 全勾配      = n
    # current     = n
    # snapshot    = n
    #
    # 合計 = 3n
    #
    # 15 outerなら
    #
    # 15 × 3 × 20242
    # = 910890
    #
    outer_iterations = 15

    test_eval_size = 10000

    # まずは理論式そのまま
    step_scale = 1.0

    # =====================================================
    # DASVRDA実行
    # =====================================================
    (
        w_opt,
        train_loss_history,
        test_acc_history,
        time_history,
        grad_count_history,
        final_test_acc
    ) = mini_batch_dasvrda(
        X_train,
        y_train,
        X_test,
        y_test,
        lam=lambda_param,
        batch_size=batch_size,
        outer_iters=outer_iterations,
        test_eval_size=test_eval_size,
        seed=42,
        step_scale=step_scale
    )

    # =====================================================
    # 結果保存
    # =====================================================
    np.savez(
        "rcv1_dasvrda_results.npz",

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
        "結果を rcv1_dasvrda_results.npz "
        "に保存しました。"
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
        "RCV1 DASVRDA: Train Loss vs Time"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "rcv1_dasvrda_time_loss.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # =====================================================
    # グラフ2
    # Loss vs Gradient Evaluations
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
        "RCV1 DASVRDA: "
        "Train Loss vs Gradient Evaluations"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "rcv1_dasvrda_gradient_loss.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # =====================================================
    # グラフ3
    # Accuracy vs Time
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
        "RCV1 DASVRDA: "
        "Test Accuracy vs Time"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "rcv1_dasvrda_time_accuracy.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()