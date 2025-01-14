import numpy as np
import re


# .raw画像を読み込み，管理するクラス
class RawReader:

    @classmethod
    def find_load(cls, fname):
        s = re.search(
            r"(\d+)(UL)?_(\d+)?[a-z]*_?SC", fname
        )  # 正規表現により画像の負荷を取得
        try:
            load = int(s.group(1))
            return load
        except ValueError as e:
            print(f"Can`t convert to int. Error:{e}")
            return load
        except TypeError as e:
            print(f"Cant`t find load. Error:{e}")

    def __init__(
        self,
        filename,  # .raw形式のデータファイル名
        min_intensity,  # 最小輝度値
        max_intensity,  # 最大輝度値
        w=1344,  # 画像データの幅
        h=1344,  # 画像データの高さ
    ):

        self.load = RawReader.find_load(filename)  # 負荷レベルを取得する
        self.w, self.h = w, h
        self.min_intensity, self.max_intensity = min_intensity, max_intensity

        # データの読み込み
        with open(filename, "rb") as f:
            self.data = np.frombuffer(f.read(), dtype=np.int16).reshape(h, w)

    # dataの正規化(画像として表示することが可能)
    def get_normalized(self):
        normalized_data = (
            (self.data - self.min_intensity)
            / (self.max_intensity - self.min_intensity)
            * 255
        )
        normalized_data = np.clip(normalized_data, 0, 255)
        return normalized_data.astype(np.uint8)
