from math import atan2
import numpy as np
import cv2
import scipy.stats as stats
import matplotlib.pyplot as plt


# 2点間の距離を返す関数(タプル対応)
def dist(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def convert2rgba(img):
    normalized_image = (img - img.min()) / (img.max() - img.min())
    rgb_image = np.stack([normalized_image] * 3, axis=-1)  # チャンネルを3倍にしてRGBに
    alpha_channel = np.ones_like(normalized_image)  # 透明度: 1.0（完全不透明）
    rgba_image = np.dstack((rgb_image, alpha_channel))  # (H, W, 4)

    return rgba_image


def convert2green(img):
    normalized_image = (img - img.min()) / (img.max() - img.min())
    zero_img = np.zeros_like(normalized_image)
    rgb_image = np.stack(
        [zero_img, normalized_image, zero_img], axis=-1
    )  # チャンネルを3倍にしてRGBに
    alpha_channel = np.ones_like(normalized_image)  # 透明度: 1.0（完全不透明）
    rgba_image = np.dstack((rgb_image, alpha_channel))  # (H, W, 4)

    return rgba_image


def combine_images(img1, img2, alpha1=0.8, alpha2=0.2):
    # 透明度を適用（アルファチャンネルを変更）
    img1[:, :, 3] = alpha1  # image1の透明度を設定
    img2[:, :, 3] = alpha2  # image2の透明度を設定

    # 画像を合成（画像のアルファチャネルで重ね合わせ）
    combined_image = np.clip(img1 * img1[:, :, 3:4] + img2 * img2[:, :, 3:4], 0, 1)

    return combined_image


# 角度を計算するメソッド
def get_angle(pts):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]

    # 主成分分析(PCA)にて方向情報を絞る
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # 先ほどのベクトル情報をもとに角度を計算
    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])

    return angle


# 曲率の絶対値の平均を計算する関数
def calc_curvature_2_derivative(x, y):

    curvatures = [0.0]
    for i in np.arange(1, len(x) - 1):
        dxn = x[i] - x[i - 1]
        dxp = x[i + 1] - x[i]
        dyn = y[i] - y[i - 1]
        dyp = y[i + 1] - y[i]
        dn = np.hypot(dxn, dyn)
        dp = np.hypot(dxp, dyp)
        dx = 1.0 / (dn + dp) * (dp / dn * dxn + dn / dp * dxp)
        ddx = 2.0 / (dn + dp) * (dxp / dp - dxn / dn)
        dy = 1.0 / (dn + dp) * (dp / dn * dyn + dn / dp * dyp)
        ddy = 2.0 / (dn + dp) * (dyp / dp - dyn / dn)
        curvature = (ddy * dx - ddx * dy) / ((dx**2 + dy**2) ** 1.5)
        curvatures.append(curvature)
    return np.mean(np.abs(curvatures))


def freedman_diaconis_rule(image):
    """Freedman-Diaconisのルールでビン幅を計算"""
    flattened = image.flatten()
    q75, q25 = np.percentile(flattened, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(flattened) ** (1 / 3))
    return max(1, int(bin_width))  # ビン幅が小さすぎないように制約


def calculate_histogram_stats(image):
    """輝度のヒストグラムに基づく統計量の計算"""
    bin_width = freedman_diaconis_rule(image)
    hist = cv2.calcHist(
        [image.astype(np.uint16)],
        [0],
        None,
        [bin_width],
        [image.min(), image.max()],
    )
    hist = hist / hist.sum()  # 正規化
    skewness = stats.skew(hist.flatten())  # 歪度（Skewness）
    kurt = stats.kurtosis(hist.flatten())  # 尖度（Kurtosis）
    return skewness, kurt


def boxplot(title, labels, arrays, y_range=(), color="lightblue", trend=False):
    # 空配列が含まれているとエラーになる
    # このため，空配列の除外と位置の指定を行う
    positions = []
    valid_arrays = []
    for i, array in enumerate(arrays, start=1):
        if len(array) > 0:
            valid_arrays.append(array)
            positions.append(i)

    if trend:
        fig, ax = plt.subplots(figsize=(7, 3))
    else:
        fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_ylabel("intensity")
    box = ax.boxplot(
        valid_arrays,
        positions=positions,
        widths=0.5,
        showmeans=True,
        meanline=True,
        patch_artist=True,
    )
    ax.set_xticks(range(1, len(labels) + 1))  # x軸の位置
    ax.set_xticklabels(labels)
    if y_range:
        ax.set_ylim(y_range[0], y_range[1] * 1.1)

    # 各要素にラベルを追加
    x_cords = []  # x座標を固定にするためのリスト
    for _, line in enumerate(box["medians"]):  # 中央値
        x, y = line.get_xdata()[1] + 0.13, line.get_ydata()[1]  # x, y座標
        x_cords.append(x)

        ax.text(x, y, f"{y:.1f}", ha="center", va="bottom", fontsize=7, color="orange")

    for i, caps in enumerate(box["caps"]):  # ひげのキャップ
        y = caps.get_ydata()[1]  # x, y座標
        ax.text(
            x_cords[i // 2],
            y,
            f"{y:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="orange",
        )

    for i, box_part in enumerate(
        box["boxes"]
    ):  # 箱の上端（第3四分位数）と下端（第1四分位数）
        box_part.set_facecolor(color)  # 色を指定
        path = box_part.get_path()  # PathPatchオブジェクトのPathを取得
        y_values = path.vertices[:, 1]  # y座標を抽出

        # 第3四分位数（上端）と第1四分位数（下端）の座標を特定
        y_upper = max(y_values)
        y_lower = min(y_values)

        # ラベルを追加
        ax.text(
            x_cords[i],
            y_upper,
            f"{y_upper:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="red",
        )
        ax.text(
            x_cords[i],
            y_lower,
            f"{y_lower:.1f}",
            ha="center",
            va="top",
            fontsize=7,
            color="red",
        )

    return fig


def violinplot(title, labels, arrays, y_range=(), color="lightblue", trend=False):
    positions = []
    valid_arrays = []
    for i, array in enumerate(arrays, start=1):
        if len(array) > 0:
            valid_arrays.append(array)
            positions.append(i)

    if len(valid_arrays) <= 0:
        return

    if trend:
        fig, ax = plt.subplots(figsize=(7, 3))
    else:
        fig, ax = plt.subplots()

    ax.set_title(title)
    ax.set_ylabel("intensity")
    vp = ax.violinplot(
        valid_arrays, positions=positions, showmedians=True, showextrema=True
    )
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")  # x軸のラベルを傾けて調整

    if y_range:
        ymax = y_range[1] * 1.3
        ax.set_ylim(y_range[0], ymax)
    else:
        ymax = np.max([np.max(arr) for arr in valid_arrays]) * 1.3
        ax.set_ylim(0, ymax)

    for pc in vp["bodies"]:
        pc.set_facecolor(color)

    # Adding labels to the median, min, and max values
    for i, array in zip(positions, valid_arrays):
        median = np.median(array)
        min_val = np.min(array)
        max_val = np.max(array)

        # x方向のオフセットを設定
        min_x = i - 0.3  # 最小値を左に
        median_x = i  # 中央値はそのまま
        max_x = i + 0.3  # 最大値を右に

        # 最小値ラベル
        ax.text(
            min_x,
            min_val,
            f"{min_val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="red",
        )

        # 中央値ラベル
        ax.text(
            median_x,
            median,
            f"{median:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="orange",
        )

        # 最大値ラベル
        ax.text(
            max_x,
            max_val,
            f"{max_val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="red",
        )

    return fig
