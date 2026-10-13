import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.datasets import load_svmlight_file


# =========================================================
# データ
# =========================================================
X, y = load_svmlight_file("rcv1_train.binary")
X = X.tocsr()

n, d = X.shape
lam = 1e-4


# =========================================================
# 目的関数
# =========================================================
def objective_and_gradient(w):

    scores = X.dot(w)

    # Logistic Loss
    tmp = -y * scores
    logistic_loss = np.mean(
        np.logaddexp(0, tmp)
    )

    regularization = (
        0.5 * lam * np.dot(w, w)
    )

    loss = (
        logistic_loss
        + regularization
    )

    # Gradient
    coeff = (
        -y
        * expit(-y * scores)
    )

    grad = (
        X.T.dot(coeff) / n
    )

    grad = np.asarray(
        grad
    ).ravel()

    grad += lam * w

    return loss, grad


# =========================================================
# L-BFGSで高精度な基準解を求める
# =========================================================
w0 = np.zeros(d)

print("Reference solutionを計算します...")
print("少し時間がかかる場合があります。")

result = minimize(
    fun=lambda w: objective_and_gradient(w),
    x0=w0,
    method="L-BFGS-B",
    jac=True,
    options={
        "maxiter": 300,
        "ftol": 1e-12,
        "gtol": 1e-8,
        "maxcor": 10
    }
)

w_ref = result.x
f_ref = result.fun

print()
print("========================================")
print("Reference Result")
print("========================================")
print("Success :", result.success)
print("Message :", result.message)
print("Iterations :", result.nit)
print(f"Reference Loss : {f_ref:.12f}")
print("========================================")


np.savez(
    "rcv1_reference.npz",
    w_ref=w_ref,
    f_ref=f_ref
)

print()
print("rcv1_reference.npz に保存しました。")