from util import convert2rgba, combine_images, convert2green, boxplot

from skimage.morphology import skeletonize
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


class Visualizer:

    # 全実験(1ex~5ex)の中の，ヒートマップの最大値
    # ヒートマップの正規化に使用
    HEATMAP_MAX = 130

    def __init__(self, extractor, fname="", save_as_png=False):
        self.ext = extractor
        (
            self.crack_dict,
            self.raw_dict,
            self.diff_dict,  # rawreaderオブジェクトではないから注意
            self.gt_dict,
            self.idimg_dict,
        ) = extractor.get_dicts()
        if fname:
            self.fname = fname
        else:
            self.fname = extractor.write_features()
        print(f"Feature File Path:{self.fname}")

        self.save_flag = save_as_png  # ファイル書き出しを行うかどうかのFlag
        if save_as_png:  # 保存に必要なディレクトリの作成
            self.__create_output_folders()

    def __create_output_folders(self):
        parent_path = "analysis"
        ex_path = self.ext.exp_num
        self.save_path = os.path.join(parent_path, ex_path)
        self.crack_path = "crack_intensity_trend"
        os.makedirs(
            os.path.join(self.save_path, self.crack_path, "fracture"), exist_ok=True
        )
        os.makedirs(
            os.path.join(self.save_path, self.crack_path, "nonfracture"), exist_ok=True
        )

    def show_skeleton(self, key, show_raw=True):
        """クラックをアノテーションした2値画像を細線化し，表示する関数．

        Args:
            key (int): 表示したい画像の負荷．
            show_raw (bool, optional): Raw画像を背景として表示するか否か. Defaults to True.
        """
        gt_img = self.gt_dict[key]  # 対応するキーのGT画像
        fracture_mask = np.copy(self.ext.fracture_mask)

        skeleton_img = skeletonize(gt_img) * 255  # 細線化を行い，np.int32に変換
        skeleton_rgba = convert2green(skeleton_img)

        # 破断面のき裂の色を変更
        indices = (fracture_mask > 0) & (skeleton_rgba[..., 1] > 0)
        skeleton_rgba[indices, 1] = 0
        skeleton_rgba[indices, 0] = 1.0

        # raw画像を背景として表示する場合
        if show_raw:
            raw_img = self.raw_dict[key].data
            raw_rgba = convert2rgba(raw_img)
            combined = combine_images(raw_rgba, skeleton_rgba, 1.0, 0.5)
        else:
            combined = skeleton_rgba

        plt.imshow(combined)
        if self.save_flag:
            cur_path = os.path.join(self.save_path, "skeleton")
            os.makedirs(cur_path, exist_ok=True)
            plt.savefig(
                os.path.join(cur_path, f"skeleton_{key}.png"),
                bbox_inches="tight",
                transparent=False,
            )
        else:
            plt.show()
        plt.close()

    def crack_total_bar(self):
        """クラック数の推移を棒グラフで表示するメソッド．
        破断面に位置するクラックをオレンジ，そうでないクラックを青で表示．"""
        df = pd.read_csv(self.fname + ".csv")
        id_list = self.ext.IDs
        ticks = list(range(1, len(id_list) + 1))

        fracture, nonfracture = [], []

        for key in id_list:
            temp = df[df["負荷"] == key]
            fracture.append(len(temp[temp["破断面"] == 1]))
            nonfracture.append(len(temp[temp["破断面"] == 0]))

        p1 = plt.bar(
            ticks, nonfracture, tick_label=id_list, align="center", color="blue"
        )
        p2 = plt.bar(
            ticks, fracture, align="center", bottom=nonfracture, color="lightcoral"
        )
        plt.ylim((0, 300))
        plt.title("Crack Total Trend")
        plt.xlabel("Load")
        plt.ylabel("Number")
        plt.legend((p1[0], p2[0]), ("Normal", "Fracture"))
        if self.save_flag:
            plt.savefig(
                os.path.join(self.save_path, "crack_total.png"),
                bbox_inches="tight",
                transparent=False,
            )
        else:
            plt.show()
        plt.close()

    def crack_area_bar(self):
        """試験片の面積に対するクラック面積の推移を棒グラフで表示するメソッド．
        破断面に位置するクラック画素をオレンジ，そうでないクラック画素を青で表示．
        """
        id_list = self.ext.IDs
        fracture_mask = self.ext.fracture_mask
        full_size = self.ext.size  # 画像サイズ

        fracture, nonfracture = [], []

        for key in id_list:
            gt_img = self.gt_dict[key]  # 対応するキーのGT画像
            fracture_s = np.sum(np.logical_and(gt_img == 255, fracture_mask == 255))
            nonfracture_s = np.sum(np.logical_and(gt_img == 255, fracture_mask == 0))

            fracture.append(fracture_s / full_size)
            nonfracture.append(nonfracture_s / full_size)

        ticks = list(range(1, len(id_list) + 1))
        p1 = plt.bar(
            ticks, nonfracture, tick_label=id_list, align="center", color="blue"
        )
        p2 = plt.bar(
            ticks, fracture, align="center", bottom=nonfracture, color="lightcoral"
        )
        plt.ylim((0, 1))
        plt.title("Crack Area Trend")
        plt.xlabel("Load")
        plt.ylabel("Area Ratio")
        plt.legend((p1[0], p2[0]), ("Normal", "Fracture"))
        if self.save_flag:
            plt.savefig(
                os.path.join(self.save_path, "crack_area.png"),
                bbox_inches="tight",
                transparent=False,
            )
        else:
            plt.show()
        plt.close()

    def diff_image_heatmap(self, rh=5, rw=5):
        """それぞれの負荷の画像の差分画像を，rh×rwのブロックで平均を取り，ヒートマップとして表示するメソッド．

        Args:
            rh (int, optional): 高さ方向のブロックサイズ．0を指定した場合は縦方向に区切らない． Defaults to 5.
            rw (int, optional): 幅方向のブロックサイズ．0を指定した場合は横方向に区切らない． Defaults to 5.
        """
        out_imgs = []
        for key in self.ext.IDs:
            diff_img = np.copy(self.diff_dict[key])
            out_img = np.zeros_like(diff_img)
            h, w = diff_img.shape
            rh = h if (rh == 0) else rh
            rw = w if (rw == 0) else rw
            for y in range(0, h, rh):
                for x in range(0, w, rw):
                    out_img[y : y + rh, x : x + rw] = np.mean(
                        diff_img[y : y + rh, x : x + rw]
                    )
            out_imgs.append(out_img)

        max_val = np.max(out_imgs)
        for i, key in enumerate(self.ext.IDs):
            plt.imshow(out_imgs[i], vmin=0, vmax=max_val, cmap="magma")
            if self.save_flag:
                cur_path = os.path.join(
                    self.save_path, "heatmap", f"intensity_{rh}_{rw}"
                )
                os.makedirs(cur_path, exist_ok=True)
                plt.savefig(
                    os.path.join(cur_path, f"heatmap_{key}.png"),
                    bbox_inches="tight",
                    transparent=False,
                )
            else:
                plt.show()
            plt.close()

    def skeleton_image_heatmap(self, rh=20, rw=20):
        """それぞれの負荷のGTのスケルトン画像を用いて，rh×rwのブロックでみたクラック数をヒートマップとして表示するメソッド．

        Args:
            rh (int, optional): 高さ方向のブロックサイズ．0を指定した場合は縦方向に区切らない． Defaults to 20.
            rw (int, optional): 幅方向のブロックサイズ．0を指定した場合は横方向に区切らない． Defaults to 20.
        """
        out_imgs = []
        for key in self.ext.IDs:
            gt_img = self.gt_dict[key]  # 対応するキーのGT画像
            skeleton_img = skeletonize(gt_img) * 255  # 細線化を行い，np.int32に変換
            out_img = np.zeros_like(skeleton_img)
            h, w = skeleton_img.shape
            rh = h if (rh == 0) else rh
            rw = w if (rw == 0) else rw
            for y in range(0, h, rh):
                for x in range(0, w, rw):
                    retval, _ = cv2.connectedComponents(
                        skeleton_img[y : y + rh, x : x + rw].astype(np.uint8),
                        connectivity=4,
                    )
                    out_img[y : y + rh, x : x + rw] = retval - 1
            out_imgs.append(out_img)

        max_val = np.max(out_imgs)
        for i, key in enumerate(self.ext.IDs):
            plt.imshow(out_imgs[i], vmin=0, vmax=max_val, cmap="magma")
            if self.save_flag:
                cur_path = os.path.join(self.save_path, "heatmap", f"total_{rh}_{rw}")
                os.makedirs(cur_path, exist_ok=True)
                plt.savefig(
                    os.path.join(cur_path, f"heatmap_{key}.png"),
                    bbox_inches="tight",
                    transparent=False,
                )
            else:
                plt.show()
            plt.close()

    def conpare_intensity_boxplot(self, key: int, crack_only=True, func=boxplot):
        """破断面とそれ以外の輝度値の箱ひげ図を表示するメソッド．

        Args:
            key (int): 表示したい画像の負荷．
            crack_only (bool, optional): クラックの輝度値のみを統計に含めるか否か．Falseの場合は，画像全体の輝度値を統計に含める． Defaults to True.
            func (function, optional): 図に使用する関数の選択．boxplotまたはviolinplot．Defaults to boxplot.
        """
        if crack_only:
            # 破断面とそうでないき裂の輝度値の箱ひげ図を表示
            fracture, nonfracture = np.array([]), np.array([])
            d_fracture, d_nonfracture = np.array([]), np.array([])

            # ※注意: 2024/12/08時点　トラッキングの関係上，同じき裂が複数のIDを持っている．
            # このため，箱ひげ図にカウントされている画素の総数が本来より多い．

            # 振り分け
            for crack in self.crack_dict.values():
                if crack.raw_img[key] is None:
                    continue
                if crack.fracture:
                    r, d = crack.raw_img[key], crack.diff_img[key]
                    fracture = np.concatenate([fracture, r[r > 0]])
                    d_fracture = np.concatenate([d_fracture, d[d > 0]])
                else:
                    r, d = crack.raw_img[key], crack.diff_img[key]
                    nonfracture = np.concatenate([nonfracture, r[r > 0]])
                    d_nonfracture = np.concatenate([d_nonfracture, d[d > 0]])
            title = "Crack Intensity Analysis"

        else:
            # 破断面とそうでない領域の輝度値の箱ひげ図を表示
            fracture_mask = np.copy(self.ext.fracture_mask)
            raw_img = self.raw_dict[key].data
            diff_img = self.diff_dict[key]

            fracture = raw_img[fracture_mask > 0]
            nonfracture = raw_img[fracture_mask == 0]
            d_fracture = diff_img[(fracture_mask > 0) & (diff_img > 0)]
            d_nonfracture = diff_img[(fracture_mask == 0) & (diff_img > 0)]

            title = "Fracture Surface Intensity Analysis"

        labels = ["fracture", "nonfracture", "diff fracture", "diff nonfracture"]
        intensity = (fracture, nonfracture, d_fracture, d_nonfracture)
        fig = func(title, labels, intensity)

        if self.save_flag:
            cur_path = os.path.join(self.save_path, "intensity", f"{func.__name__}")
            if not crack_only:
                cur_path = os.path.join(cur_path, "surface")
            os.makedirs(cur_path, exist_ok=True)
            fig.savefig(
                os.path.join(
                    cur_path,
                    f"intensity_{key}{'_crack_only' if crack_only else ''}.png",
                ),
                bbox_inches="tight",
                transparent=False,
            )
        else:
            fig.show()
        plt.close()

    def conpare_intensity_trend_boxplot(self, use_diff=False, func=boxplot):
        """破断面とそれ以外の輝度値の推移を箱ひげ図で比較するメソッド．

        Args:
            func (function, optional): 図に使用する関数の選択．boxplotまたはviolinplot．Defaults to boxplot.
        """

        id_list = self.ext.IDs
        fracture, nonfracture = [], []

        for key in id_list:

            # 破断面とそうでない領域の輝度値の箱ひげ図を表示
            fracture_mask = np.copy(self.ext.fracture_mask)
            if use_diff:
                raw = self.diff_dict[key]
                fracture.append(raw[(fracture_mask > 0) & (raw > 0)])
                nonfracture.append(raw[(fracture_mask == 0) & (raw > 0)])
            else:
                raw = self.raw_dict[key].data
                fracture.append(
                    raw[
                        (fracture_mask > 0)
                        & (raw > self.ext.min)
                        & (raw < self.ext.max)
                    ]
                )
                nonfracture.append(
                    raw[
                        (fracture_mask == 0)
                        & (raw > self.ext.min)
                        & (raw < self.ext.max)
                    ]
                )

        if use_diff:
            y_range = (0, self.ext.diff_max)
        else:
            y_range = (self.ext.min, self.ext.max)

        cur_path = os.path.join(self.save_path, "intensity")
        os.makedirs(cur_path, exist_ok=True)

        ffig = func(
            "Fracture Surface Intensity Trend Analysis",
            id_list,
            fracture,
            y_range,
            "lightcoral",
            trend=True,
        )

        ffig.savefig(
            os.path.join(
                cur_path,
                f"fracture_intensity_trend{'_diff' if use_diff else ''}.png",
            ),
            bbox_inches="tight",
            transparent=False,
        )
        plt.close(ffig)

        nfig = func(
            "Nonfracture Surface Intensity Trend Analysis",
            id_list,
            nonfracture,
            y_range,
            trend=True,
        )

        nfig.savefig(
            os.path.join(
                cur_path,
                f"nonfracture_intensity_trend{'_diff' if use_diff else ''}.png",
            ),
            bbox_inches="tight",
            transparent=False,
        )
        plt.close(nfig)

    def crack_intensity_trend_boxplot(self, use_diff=False, func=boxplot):
        """クラック単位で輝度値の推移を箱ひげ図で表示するメソッド．

        Args:
            use_diff (bool, optional): 輝度値の計算をする際に，Raw画像の代わりに，ベースとの差分画像を用いるか否か. Defaults to False.
            func (function, optional): 図に使用する関数の選択．boxplotまたはviolinplot．Defaults to boxplot.
        """
        # 現在は破断面に位置しているクラックのみを順に表示している．
        # 今後はID指定の機能を追加．
        # 破断面以外のクラックも表示するとめんどいので，すべてのクラックの図をPDFに書き出す機能．
        # 全部の箱ひげ図を見るのはめんどいので，増加率的なものを数値化すると良い？
        id_list = self.ext.IDs
        if use_diff:
            y_range = (0, self.ext.diff_max)
        else:
            y_range = (self.ext.min, self.ext.max)

        for crack in self.crack_dict.values():
            if crack.fracture:  # 破断き裂を赤
                color = "lightcoral"
                group = "fracture"
            else:
                color = "lightblue"
                group = "nonfracture"
            title = f"Crack {crack.id} Intensity Trend"
            raw_imgs = ()
            for key in id_list:
                if use_diff:
                    raw = crack.diff_img[key]
                else:
                    raw = crack.raw_img[key]
                if raw is None:
                    raw_imgs += (np.array([]),)  # 空配列を追加
                    continue
                raw_imgs += (raw[raw > 0],)
            fig = func(title, id_list, raw_imgs, y_range, color, trend=True)
            if not fig:
                continue

            if self.save_flag:
                fig.savefig(
                    os.path.join(
                        self.save_path,
                        self.crack_path,
                        group,
                        f"intensity_trend_{crack.id}.png",
                    ),
                    bbox_inches="tight",
                    transparent=False,
                )
            else:
                fig.show()
            plt.close()

    def conpare_crack_shape_boxplot(self, key: int = None, func=boxplot):
        """破断面とそうでないクラックの，形状に関する特徴量を箱ひげ図で比較するメソッド．

        Args:
            key (int, optional): 表示したい画像の負荷．指定がない場合，全負荷を計算する． Defaults to None.
            func (function, optional): 図に使用する関数の選択．boxplotまたはviolinplot．Defaults to False.
        """
        # 形状に関する特徴量の箱ひげ図比較
        df = pd.read_csv(self.fname + ".csv")
        fracture_df = df[df["破断面"] == 1]
        nonfracture_df = df[df["破断面"] == 0]

        items = {
            "面積": "Area",
            "周囲長": "Perimeter",
            "長さ": "Length",
            "太さ": "Thickness",
            "コンパクト比": "Compactness",
            "凹凸係数": "Roughness",
        }

        labels = ["fracture", "nonfracture"]
        for item in items:
            y_range = (0, df[item].max())
            if key:
                k_fracture_df, k_nonfracture_df = (
                    fracture_df[fracture_df["負荷"] == key],
                    nonfracture_df[nonfracture_df["負荷"] == key],
                )
                raw_imgs = (k_fracture_df[item], k_nonfracture_df[item])
            else:
                raw_imgs = (fracture_df[item], nonfracture_df[item])

            title = f"{items[item]} Trend"
            fig = func(title, labels, raw_imgs, y_range)

            if self.save_flag:
                cur_path = os.path.join(self.save_path, items[item].lower())
                os.makedirs(cur_path, exist_ok=True)
                fig.savefig(
                    os.path.join(
                        cur_path,
                        f"crack_{items[item].lower()}{'_' + str(key) if key else ''}.png",
                    ),
                    bbox_inches="tight",
                    transparent=False,
                )
            else:
                fig.show()
            plt.close()

    def crack_shape_trend_boxplot(self, func=boxplot):
        """クラックの形状に関する特徴量の推移を箱ひげ図で表示するメソッド．

        Args:
            func (function, optional): 図に使用する関数の選択．boxplotまたはviolinplot．Defaults to False.
        """
        # 形状に関する特徴量の箱ひげ図比較
        df = pd.read_csv(self.fname + ".csv")
        id_list = self.ext.IDs  # 負荷リスト

        items = {
            "面積": "Area",
            "周囲長": "Perimeter",
            "長さ": "Length",
            "太さ": "Thickness",
            "コンパクト比": "Compactness",
            "凹凸係数": "Roughness",
        }

        types = ["Nonfracture", "Fracture"]
        for item in items:
            y_range = (0, df[item].max())
            for i, t in enumerate(types):
                color = "lightblue" if i == 0 else "lightcoral"
                plot_items = ()
                for key in id_list:
                    temp = df[(df["破断面"] == i) & (df["負荷"] == key)]
                    plot_items += (temp[item],)

                title = f"{t} Crack {items[item]} Trend"
                fig = func(title, id_list, plot_items, y_range, color, trend=True)

                if self.save_flag:
                    cur_path = os.path.join(self.save_path, items[item].lower())
                    os.makedirs(cur_path, exist_ok=True)
                    fig.savefig(
                        os.path.join(
                            cur_path,
                            f"shape_trend_{items[item].lower()}_{t.lower()}.png",
                        ),
                        bbox_inches="tight",
                        transparent=False,
                    )
                else:
                    fig.show()
                plt.close()


if __name__ == "__main__":
    # main.main()
    pass
