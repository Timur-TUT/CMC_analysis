from rawreader import RawReader
from crack import Crack
from util import *

from matplotlib import patches
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import glob
import os
import csv
import cv2
import random
import datetime
import h5py


class Extractor:

    DIST_TH = 150  # き裂の重心間の距離を調べる際の閾値
    IoU_TH = 0.7  # き裂の重なりを調べる際の閾値
    NOISE_TH = 20  # 差分画像の残ったノイズを削除する閾値
    ID_dict = {
        "1ex": tuple(i for i in range(1200, 3000, 200)),
        "3ex": (2800,) + tuple(i for i in range(3200, 4200, 200)),
        "4ex": tuple(i for i in range(1600, 3600, 200)),
        "5ex": (1800,) + tuple(i for i in range(2000, 2700, 100)),
    }  # 各実験の分析対象負荷の辞書
    color = {
        k: tuple([random.random() for _ in range(3)]) for k in range(300)
    }  # 表示用カラー

    def __init__(self, input_folder, exp_num):
        self.__crack_dict = {}  # 全てのき裂を保管する辞書
        self.__raw_dict = {}  # raw画像辞書
        self.__diff_dict = {}  # 差分画像辞書
        self.__gt_dict = {}  # gt画像辞書
        self.__idimg_dict = {}  # idマップの辞書

        self.exp_num = exp_num
        self.IDs = Extractor.ID_dict[exp_num[:3]]  # 実験に対応する負荷のtupleを定義
        path = os.path.join(input_folder, exp_num)
        file_names = sorted(
            glob.glob(f"{path}/*.raw"), key=lambda x: RawReader.find_load(x)
        )

        # 破断面をアノテーションした画像の読み込み
        self.fracture_mask = cv2.imread(
            f"gt/{exp_num[:3]}/fracture_gt.png", cv2.IMREAD_GRAYSCALE
        )
        if self.fracture_mask is None:
            self.fracture_coords = set()
            print("Warning: no fracture mask")
        else:
            y_array, x_array = np.where(self.fracture_mask > 0)
            self.fracture_coords = set(
                [(y, x) for (y, x) in zip(y_array, x_array)]
            )  # 破断面の座標を取得

        # 必要なパラメータをparameters.txtから読み込む
        # self.w, self.h, self.min, self.maxが定義される
        self.read_params(path)
        self.params = (self.min, self.max, self.w, self.h)
        self.diff_max = 20

        # NLMeansがかかったベース画像の読み込み．※base画像の負荷が一番小さい前提
        self.base_obj = RawReader(
            f"{path}/{os.path.basename(file_names[0])}", *self.params
        )

        for fname in file_names:
            fname = os.path.basename(fname)
            try:
                load = int(fname.split("_")[0])
            except ValueError as e:
                print(f"Error: {e}. \nSkipping '{fname}'")
                continue

            if load not in self.IDs:
                continue

            # raw画像の読み込み
            raw_obj = RawReader(f"{path}/{fname}", *self.params)
            self.__raw_dict[load] = raw_obj

            # 対応するGT画像の読み込み
            gt_img = cv2.imread(f"gt/{exp_num[:3]}/{load}_gt.png", cv2.IMREAD_GRAYSCALE)
            self.__gt_dict[load] = gt_img

            self.create_instances(raw_obj.data, gt_img, load)

    def get_dicts(self):
        return (
            self.__crack_dict.copy(),
            self.__raw_dict.copy(),
            self.__diff_dict.copy(),
            self.__gt_dict.copy(),
            self.__idimg_dict.copy(),
        )

    def read_params(self, path):
        with open(
            os.path.join(path, "parameters.txt"),
            mode="r",
            encoding="utf-8",
        ) as f:
            for line in f:
                line = line.strip()  # 前後の空白や改行を削除
                if "=" in line:
                    exec(line)

    def create_instances(self, rimg, limg, load):
        idx = self.IDs.index(load)  # IDsリスト中のインデックス番号
        id_map = np.zeros_like(limg)  # id情報のマップ
        fracture_map = np.zeros_like(limg)  # 破断き裂マップ(可視化にのみ使用)

        diff_img = rimg - self.base_obj.data
        diff_img[diff_img < Extractor.NOISE_TH] = 0  # ノイズ部分を考慮しない
        self.__diff_dict[load] = diff_img
        if np.max(diff_img) > self.diff_max:  # diff画像の最大値を記憶
            self.diff_max = np.max(diff_img)

        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
            limg, connectivity=4
        )  # 個数, ラベリング画像, 領域情報(x,y,w,h,s), 重心(x,y)
        sorted_indices = sorted(
            range(1, retval), key=lambda i: (centroids[i][1], centroids[i][0])
        )  # IDをラスタ順にするためのsort
        for i in sorted_indices:  # 背景以外を除き，全てのき裂を解析
            stat, centroid = stats[i], tuple(centroids[i])  # 領域情報, 重心
            bb = [
                stat[0],
                stat[1],
                stat[0] + stat[2],
                stat[1] + stat[3],
            ]  # バウンディングボックス(x,y,x+w,y+h)
            rim = np.copy(rimg[bb[1] : bb[3], bb[0] : bb[2]])  # raw画像
            lim = np.copy(labels[bb[1] : bb[3], bb[0] : bb[2]])  # masking画像
            dim = np.copy(diff_img[bb[1] : bb[3], bb[0] : bb[2]])  # 差分画像
            lim[lim != i] = 0  # き裂ではない部分を0埋め
            rim[lim != i] = 0
            dim[lim != i] = 0
            y_array, x_array = np.where(
                labels == i
            )  # 画像における着目き裂の画素の座標を取得
            coordinates_set = set(
                [(y, x) for (y, x) in zip(y_array, x_array)]
            )  # き裂の画素座標を集合に
            if len(self.fracture_coords) == 0:  # 破断面のアノテーション画像が無い場合
                fracture = None
            elif (
                len(coordinates_set & self.fracture_coords) / len(coordinates_set)
                >= 0.1
            ):  # き裂一割以上が破断面に位置するなら1
                fracture = 1
            else:  # それ以外は0
                fracture = 0

            merged_list = []
            for v in self.__crack_dict.values():  # 同じIDのき裂がないかを探索
                prev_coordinates_set = v.coordinates[self.IDs[idx - 1]]
                if prev_coordinates_set:
                    if (
                        dist(centroid, v.centroid[self.IDs[idx - 1]])
                        > Extractor.DIST_TH
                    ):  # ユークリッド距離が遠すぎるものでは実施しない
                        continue
                    iou = len(coordinates_set & prev_coordinates_set) / len(
                        prev_coordinates_set
                    )  # 重なる面積を計算 (以前のき裂の画素の何割が共通か)
                    if iou >= Extractor.IoU_TH:  # 更新
                        v.add(
                            load,
                            rim,
                            lim,
                            dim,
                            bb,
                            stat[4],
                            centroid,
                            coordinates_set,
                        )
                        v.fracture = fracture
                        merged_list.append(v)
                        id_map[y_array, x_array] = v.id
                        fracture_map[y_array, x_array] = (
                            v.fracture
                        )  # 破断き裂マップ(可視化にのみ使用)

            if len(merged_list) == 0:  # それ以外は新規のき裂オブジェクト
                obj = Crack(self.IDs)
                obj.add(load, rim, lim, dim, bb, stat[4], centroid, coordinates_set)
                obj.fracture = fracture
                self.__crack_dict[obj.id] = obj
                id_map[y_array, x_array] = obj.id
                fracture_map[y_array, x_array] = (
                    obj.fracture
                )  # 破断き裂マップ(可視化にのみ使用)
            elif len(merged_list) > 1:  # 結合したき裂の場合はそのIDを登録
                ids = tuple(sorted([v.id for v in merged_list]))
                for v in merged_list:
                    v.merged[load] = ids
        self.__idimg_dict[load] = id_map

        # 破断き裂マップの可視化(必要に応じてコメントアウトを外す)
        # _, ax = plt.subplots(3, 1)
        # ax[0].imshow(limg)
        # ax[1].imshow(self.fracture_mask)
        # ax[2].imshow(fracture_map)
        # plt.show()

    def write_features(self):
        # CSV形式で形状等の情報，HDF5形式でき裂の画像をファイルに書き込み
        dt_now = datetime.datetime.now()
        dt_text = dt_now.strftime("%y%m%d%H%M%S")
        fname = f"out/crack_feature_{self.exp_num}_{dt_text}"

        with open(
            fname + ".csv", "w", newline="", encoding="utf-8-sig"
        ) as fc, h5py.File(fname + ".hdf5", mode="w") as fh:
            writer = csv.writer(fc)
            title = [
                "ID",
                "負荷",
                "結合ID",
                "破断面",  # 破断面に位置するき裂は1，それ以外は0と出力
                "重心(x)",
                "重心(y)",
                "面積",
                "方向",
                "周囲長",
                "曲率平均",  # 各点の曲率を計算した絶対値平均
                "長さ",  # 縦方向の長さ
                "太さ",  # 横方向の長さ
                "コンパクト比",  # 周囲長 ÷ ( 2 × √(π × 面積) ) ⇒ 円のときに1, 円から離れると大きくなる
                "凹凸係数",  # 周囲長^2 ÷ 面積 × (π/4) ⇒ 円のとき9.85程度, 凹凸が多いと値が大きくなる
            ]
            writer.writerow(title)
            for crack in tqdm(self.__crack_dict.values()):
                for key in self.IDs:
                    img = crack.raw_img[key]
                    if img is not None:
                        # 形状情報の書き込み(CSV形式)
                        s = crack.s[key]
                        row = [
                            crack.id,
                            key,
                            crack.merged[key],
                            crack.fracture,
                            *crack.centroid[key],
                            s,
                        ]
                        bin_img = ((img > 0) * 255).astype(np.uint8)  # 二値化画像
                        # 輪郭の抽出
                        contours, _ = cv2.findContours(
                            bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                        )
                        if len(contours) != 1:
                            continue

                        for contour in contours:
                            # き裂の主要方向
                            angle = get_angle(contour)

                            # 周囲長
                            perimeter = cv2.arcLength(contour, True)

                            # 曲率平均
                            curv = calc_curvature_2_derivative(
                                contour[:, 0, 0], contour[:, 0, 1]
                            )
                            row += [angle, perimeter, curv]

                            # 回転矩形
                            rect = cv2.minAreaRect(contour)
                            box = cv2.boxPoints(rect)
                            box = np.int0(box)

                            # 長さと幅
                            length = max(rect[1])
                            width = min(rect[1])
                            row += [length, width]

                            # コンパクト比
                            compactness = perimeter / (2 * np.sqrt(np.pi * s))
                            # 凹凸係数
                            roughness = perimeter**2 / s * np.pi / 4
                            row += [compactness, roughness]

                            writer.writerow(row)

                        # 輝度値情報の書き込み(HDF5形式)
                        try:
                            crack_group = fh.create_group(
                                f"{crack.id}"
                            )  # IDごとにグループを作成
                        except ValueError:
                            crack_group = fh[f"{crack.id}"]  # 登録済みのき裂
                        load_group = crack_group.create_group(
                            f"{key}"
                        )  # 負荷のグループを作成
                        load_group.create_dataset(
                            "重心", data=crack.centroid[key]
                        )  # タプル
                        load_group.create_dataset(
                            "破断面", data=crack.fracture
                        )  # None,0,1
                        try:
                            load_group.create_dataset(
                                "結合ID", data=crack.merged[key]
                            )  # タプル
                        except TypeError:
                            load_group.create_dataset("結合ID", data=())  # タプル
                        load_group.create_dataset(
                            "領域座標", data=crack.bb[key]
                        )  # タプル
                        load_group.create_dataset("画像", data=crack.raw_img[key])
        return fname

    # ※要修正．未リファクタリング
    def show(self, cmc, key, id=None, enable_rect=False):
        _, ax = plt.subplots(1, tight_layout=True)
        ax.imshow(cmc.data, cmap="gray")
        ax.set_title("Labeled Image")
        crack_list = list(self.__crack_dict.values())

        # 結果表示
        for i in range(len(crack_list)):
            if (id != None and crack_list[i].id != id) or (
                crack_list[i].bb[key] is None
            ):
                continue
            x, y, width, height = crack_list[i].bb[key]  # x座標, y座標, 幅, 高さ, 面積

            if enable_rect:
                rect = patches.Rectangle(
                    (x - 1, y - 1),
                    (width + 1) - x,
                    (height + 1) - y,
                    linewidth=2,
                    edgecolor=Extractor.color[crack_list[i].id],
                    facecolor="none",
                )
                ax.add_patch(rect)
            else:
                # バウンディングボックス内に対応する画像を貼り付け
                bg = np.copy(cmc.data)
                crack_image = crack_list[i].raw_img[key]
                if crack_image is not None:
                    a = bg[y : y + crack_image.shape[0], x : x + crack_image.shape[1]]
                    a[crack_image > 0] = np.max(bg) * 1.1
                ax.imshow(bg, cmap="gray")

            # バウンディングボックスの右下にIDを表示
            ax.text(
                width + 1,
                height + 1,
                f"ID:{crack_list[i].id}",
                color=Extractor.color[crack_list[i].id],
                fontsize=12,
            )
        plt.show()


if __name__ == "__main__":
    ext = Extractor("4ex_1")
    ext.write_features()
