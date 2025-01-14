class Crack:

    crack_num = 1  # 全ての画像を通してのき裂の数(IDに使用)

    def __init__(self, IDs):
        self.id = Crack.crack_num  # き裂のID(発生した順番に付与する)
        Crack.crack_num += 1

        adict = {k: None for k in IDs}  # ダミー
        self.raw_img = (
            adict.copy()
        )  # バウンディングボックスで切り抜かれた生データの画像
        self.label_img = (
            adict.copy()
        )  # rawデータの中でもき裂に当たる部分がラベリングされた画像
        self.diff_img = (
            adict.copy()
        )  # rawデータの中でもき裂に当たる部分がラベリングされた画像
        self.bb = adict.copy()  # 画像全体の中でのバウンディングボックスの角の座標
        self.s = adict.copy()  # 面積
        self.centroid = adict.copy()  # 重心座標
        self.coordinates = adict.copy()  # 座標の集合
        self.merged = adict.copy()  # 結合したき裂のIDを保存
        self.fracture = None  # 破断面に位置するき裂は1，それ以外は0

    # 同じIDのき裂を追加するメソッド
    def add(self, key, raw_img, label_img, diff_img, bb, s, centroid, coordinates):
        self.raw_img[key] = raw_img
        self.label_img[key] = label_img
        self.diff_img[key] = diff_img
        self.bb[key] = bb
        self.s[key] = s
        self.centroid[key] = centroid
        self.coordinates[key] = coordinates
