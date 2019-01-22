import ridge_regression as ridge
import linear_regression as linearreg
import numpy as np
import matplotlib.pyplot as plt


# データの設定
x = np.arange(12)
y = 1 + 2 * x

# 外れ値の設定
y[2] = 20
y[4] = 0

xmin = 0
xmax = 12
ymin = -1
ymax = 25
# ゴールとしては、一行目に普通の線形回帰、二行目にリッジ回帰で予測したマップを表示させる。それをデータ量に分けて横に五つ表示。
fig, axes = plt.subplots(nrows=2, ncols=5)  # 縦横の数を設定し、そのfigインスタンスとaxesインスタンスを作成。
for i in range(5):
    axes[0, i].set_xlim([xmin, xmax])
    axes[0, i].set_ylim([ymin, ymax])
    axes[1, i].set_ylim([ymin, ymax])
    axes[1, i].set_ylim([ymin, ymax])

    # 徐々にサンプル数を増やしたいため、iの値によってデータをスプリット
    xx = x[:2 + i * 2]
    yy = y[:2 + i * 2]
    # 一行目、二行目ともに同じデータを散文図で設定。
    axes[0, i].scatter(xx, yy, color="k")
    axes[1, i].scatter(xx, yy, color="k")

    # 普通の線形回帰
    model = linearreg.LinearRegression()
    model.fit(xx, yy)
    # 線形の図の始端、終端を定義
    xs = [xmin, xmax]
    ys = [model.w_[0] + model.w_[1] * xmin,
          model.w_[0] + model.w_[1] * xmax]
    # 図示するため0行目のi列に代入
    axes[0, i].plot(xs, ys, color="k")

    # リッジ回帰
    model = ridge.RidgeRegression(lambda_=10.)
    model.fit(xx, yy)
    xs = [xmin, xmax]
    ys = [model.w_[0] + model.w_[1] * xmin,
          model.w_[0] + model.w_[1] * xmax]
    axes[1, i].plot(xs, ys, color="k")

plt.show()

"""
図でわかる通り、リッジ回帰ではサンプル数が少ない時に、例外的なデータからの影響を受けにくい、という性質がある。
これが正則化項wによる効果。
これが望ましいかどうかは、適用しようとしているデータの性質と、「何に応用しているのか」による。
"""