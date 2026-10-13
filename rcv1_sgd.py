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

    # Sparse形式のまま読み込む
    X_train, y_train = load_svmlight_file(
        train_file
    )

    X_test, y_test = load_svmlight_file(
        test_file,
        n_features=X_train.shape[1]
    )

    # CSR形式にしておく
    X_train = X_train.tocsr()
    X_test = X_test.tocsr()

    return X_train, y_train, X_test, y_test


# =========================================================
# 2. 目的関数
#    L2正則化付きロジスティック回帰
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
# 4. ミニバッチ勾配
# =========================================================
def mini_batch_gradient(
    X_batch,
    y_batch,
    w,
    lam
):

    # wでの予測スコア
    scores = X_batch.dot(w)

    # logistic gradientの係数
    #
    # -y / (1 + exp(y * score))
    #
    coeff = (
        -y_batch
        * expit(
            -y_batch * scores
        )
    )

    # バッチ平均勾配
    grad = (
        X_batch.T.dot(coeff)
        / X_batch.shape[0]
    )

    # ndarrayに変換
    grad = np.asarray(
        grad
    ).ravel()

    # L2正則化
    grad += lam * w

    return grad


# =========================================================
# 5. Mini-batch SGD
# =========================================================
def mini_batch_sgd(
    X_train,
    y_train,
    X_test,
    y_test,
    lam=1e-4,
    learning_rate=0.1,
    batch_size=128,
    epochs=45,
    test_eval_size=10000,
    seed=42
):

    np.random.seed(seed)

    n, d = X_train.shape

    # 初期値
    w = np.zeros(d)

    # -----------------------------------------
    # 評価用テストデータ
    # 毎epoch全67万件評価すると重いため
    # 最初の10000件を使用
    # -----------------------------------------
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

    # -----------------------------------------
    # 履歴
    # -----------------------------------------
    train_loss_hist = []
    test_acc_hist = []
    time_hist = []
    grad_count_hist = []

    optimization_time = 0.0
    grad_count = 0

    # -----------------------------------------
    # 条件表示
    # -----------------------------------------
    print()
    print("========================================")
    print("RCV1 Mini-batch SGD")
    print("========================================")
    print(f"Train samples   : {n}")
    print(f"Features        : {d}")
    print(f"Batch size      : {batch_size}")
    print(f"Epochs          : {epochs}")
    print(f"lambda          : {lam}")
    print(f"Learning rate   : {learning_rate}")
    print(f"Test eval size  : {test_eval_size}")
    print("========================================")
    print()

    # -----------------------------------------
    # 初期状態
    # -----------------------------------------
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
    print("学習開始...")
    print()

    # =====================================================
    # Epochループ
    # =====================================================
    for epoch in range(epochs):

        start_time = time.perf_counter()

        # 訓練データの順番をランダムに並べる
        indices = np.random.permutation(n)

        # -----------------------------------------
        # Mini-batchループ
        # -----------------------------------------
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

            # 勾配
            grad = mini_batch_gradient(
                X_batch,
                y_batch,
                w,
                lam
            )

            # SGD更新
            w = (
                w
                - learning_rate * grad
            )

            # 使用したデータ数を勾配評価回数としてカウント
            grad_count += len(
                batch_indices
            )

        end_time = time.perf_counter()

        optimization_time += (
            end_time
            - start_time
        )

        # -----------------------------------------
        # Epoch終了後の評価
        # -----------------------------------------
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
            f"Epoch {epoch + 1:2d}/{epochs} | "
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

    # 最後だけRCV1の全Testデータを使用
    print()
    print("全Testデータで最終Accuracyを計算中...")

    final_test_acc_full = compute_accuracy(
        X_test,
        w,
        y_test
    )

    print()
    print("========================================")
    print("Final Result")
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
# 6. メイン処理
# =========================================================
if __name__ == "__main__":

    # -----------------------------------------
    # ファイル名
    # -----------------------------------------
    train_file = "rcv1_train.binary"
    test_file = "rcv1_test.binary"

    # -----------------------------------------
    # データ読み込み
    # -----------------------------------------
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

    # -----------------------------------------
    # ハイパーパラメータ
    # -----------------------------------------
    lambda_param = 1e-4

    learning_rate = 0.1

    batch_size = 128

    # a9aと同じく
    # 45 epochから試す
    epochs = 45

    # 毎epochのAccuracy評価には
    # Testデータ10000件だけ使用
    test_eval_size = 10000

    # -----------------------------------------
    # 実行
    # -----------------------------------------
    (
        w_opt,
        train_loss_history,
        test_acc_history,
        time_history,
        grad_count_history,
        final_test_acc
    ) = mini_batch_sgd(
        X_train,
        y_train,
        X_test,
        y_test,
        lam=lambda_param,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        test_eval_size=test_eval_size,
        seed=42
    )

    # =====================================================
    # 結果を保存
    # 後でSVRG/DASVRDAと重ねるため
    # =====================================================
    np.savez(
        "rcv1_sgd_results.npz",
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
        "結果を rcv1_sgd_results.npz に保存しました。"
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
        "RCV1 SGD: Train Loss vs Time"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "rcv1_sgd_time_loss.png",
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
        "RCV1 SGD: Train Loss vs Gradient Evaluations"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "rcv1_sgd_gradient_loss.png",
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
        "RCV1 SGD: Test Accuracy vs Time"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "rcv1_sgd_time_accuracy.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()