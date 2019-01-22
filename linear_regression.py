import numpy as np
from scipy import linalg

"""
■特徴量ベクトルが多次元の場合
もっと一般化し、特徴量ベクトルがd次元である時を考える。
この場合、前述のように特徴量行列はn×d行列Xで表される。

一般次元の場合の線形回帰モデルは次のような式で表される。

y = w0 +  w1*x1 * w2*x2 + ...+ wd*xd + ε  ...(5-5)

ここで、
(x0, ..., xd).Tは入力変数(一個一個が多次元のベクトル)
w0, w1, ..., wdはパラメタ
yはターゲット
εはノイズを示す。
特に、d=1の時は前述の1次元のケースに相当し、
w0がb, w1がaに対応する。

前にも説明したが、
x = (x1, ... , xd).T
に対し要素1を蒸したベクトルxtil(1, x1, ..., xd)を考え、ベクトルw=(w0, ...wd).T
を定義すると以下のように表せられる。

y = w.Txtil 

次にこのモデルを実際のデータに当てはめてみる。
Xに対応するXtilを考えると、この時の当てはめは次のように表せられる。

^y(w) = Xtil*w

この当てはめと、ターゲットyとのさの２乗の和|| y - ^y(w) ||^2を最小化することを考える。
つまり、次の値を最小化するようなwを求める。

E(w) = ||y-Xtil*w||^2
     = (y - Xtil*w).T * (y - Xtil*w)
     = y.T*y - w.T*Xtil.T*y - y.T*Xtil*w + w.T*Xtil.T * Xtil*w   # 行列なので、第二、第三項目は足せない。

この勾配を計算する

∇E(w) = -2Xtil*y + 2Xtil.T*Xtil*w

これを=0と置くことで、最小化するwが求められる

Xtil*y = Xtil.T*Xtil*w
   w = (Xtil.T*Xtil)^-1 * Xtil.T*y

では、これを実装してみる。先ずは回帰を計算するクラスだけを実装する。
"""


class LinearRegression:
    """
    Xtil*y = Xtil.T*Xtil*w
   w = (Xtil.T*Xtil)^-1 * Xtil.T*y
    =>
    Xtil*y = Xtil.T*Xtil*W
    w = (Xtil.T*Xtil)^-1 * Xtil.T*y
    """
    def __init__(self):
        self.w_ = None

    def fit(self, X, t):
        """
        訓練データによる学習を行い、self.w_に保存していくメソッド

        a (x y z) = b
        このaとbの方程式を満たすような一次元配列(x,y,z)を求めたいとき、np.linalg.solveというのがある
        np.linalg.solve(a, b)

        これを使い、w = (Xtil.T*Xtil)^-1 * Xtil.T*y を求める。
        :param X: 入力訓練データ
        :param t: 出力訓練データ
        :return:
        """
        Xtil = np.c_[np.ones(X.shape[0]), X]  # Xtilのこと。つまり、入力行列の左端に1を付け加えている。オフセット
        A = np.dot(Xtil.T, Xtil)  # 第1項目。invじゃなくて良いの？と思うが、linalg.solveの形によりこのままで良い。(下で説明)
        b = np.dot(Xtil.T, t)  # 第2項目
        # これらのwにおける一次方程式をlinalg.solveを使って解きたいが、
        # AX = B <=> Aw = b の形でなくてはならないので、
        # w = (Xtil.T*Xtil)^-1 * Xtil.T*y <=> Xtil.T*Xtil*W = Xtil*y <=> Aw = b
        self.w_ = linalg.solve(A, b)

    def predict(self, X):
        """
        入力値に対し出力値を予想するメソッド。
        学習時と同様、Xの各行をサンプルとして、それぞれについて予測。
        :param X:
        :return:
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)  # ちゃんと一次元に直す。
        Xtil = np.c_[np.ones(X.shape[0]), X]  # 初項が係数の1が含まれたXを作成。(woと計算するため)
        return np.dot(Xtil, self.w_)  # 学習したwとの内積を取り、多次元関数を作成。これがXそれぞれのデータと最も近い形の関数となる。


if __name__ == "__main__":
    pass
