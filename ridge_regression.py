
"""
■リッジ回帰
線形回帰で最小化する目的関数に、パラメータの大きさの項を足したのがリッジ回帰。
つまり、リッジ回帰では次の関数を最小化するようなwを決定する。

E(w) = ||y - Xtil*w||^2 + λ||w||^2

λ||w||^2 の項(正則化項と呼ばれる)が加わったことにより、
点群を線形に近似しつつも「できるだけwの大きさ(L2ノルム)が小さい方が良い」という力が働く。(そうしないとEが大きくなってしまう。)
ここでλは「wの大きさをどのくらい重視するか」を表す定数で、ハイパーパラメータ と呼ばれる。

では、線形回帰の時と同様にwについての勾配をとって=0と置く

∇E = 2Xtil.T*Xtil*w - 2Xtil.T*y + 2λ*w
   = 2[(Xtil.T*Xtil + λI)*w - Xtil.T*y] = 0  # Iは単位行列
寄って

(Xtil.T*Xtil + λI)*w = Xtil.T*y  # (A + λc)*X = b
w  = (Xtil.T*Xtil + λ*I)^-1 * Xtil.T*y

これを実装し、線形回帰と同様のデータで検証してみる。
"""

import numpy as np
from scipy import linalg


class RidgeRegression:

    def __init__(self, lambda_=1.):
        self.lambda_ = lambda_  # λ
        self.w_ = None

    def fit(self, X, t):
        """
        linalg.solver(a,b)を使うので、 (A + λc)X = bの形に変更
        :param X:
        :param t:
        :return:
        """
        Xtil = np.c_[np.ones(X.shape[0]), X]
        c = np.eye(Xtil.shape[1])  # 単位行列Iを(Xと同じ次元で)生成
        A = np.dot(Xtil.T, Xtil) + self.lambda_ * c
        b = np.dot(Xtil.T, t)
        self.w_ = linalg.solve(A, b)

    def predict(self, X):
        Xtil = np.c_[np.ones()]
        return np.dot(Xtil, self.w_)
