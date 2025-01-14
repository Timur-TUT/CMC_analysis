from extractor import Extractor
from visualizer import Visualizer
from util import boxplot, violinplot

import time


def main():
    ### 全体をまとめて実行す
    for ex_name in ["1ex", "3ex", "4ex_1", "5ex"]:
        print(f"\nSTARTING {ex_name} ANALYSIS\n")
        input_folder = "filtered_more"
        ext = Extractor(input_folder, ex_name)
        vis = Visualizer(ext, save_as_png=True)
        vis.crack_total_bar()
        vis.crack_area_bar()
        print("\nINTENSITY TREND: START\n")
        vis.crack_intensity_trend_boxplot(use_diff=True, func=violinplot)
        print("\nINTENSITY TREND: DONE\n")
        vis.conpare_intensity_trend_boxplot(use_diff=True, func=violinplot)
        vis.conpare_intensity_trend_boxplot(use_diff=False, func=violinplot)
        vis.conpare_crack_shape_boxplot(func=violinplot)
        vis.crack_shape_trend_boxplot(func=violinplot)
        vis.diff_image_heatmap(10, 10)
        vis.diff_image_heatmap(15, 15)
        vis.diff_image_heatmap(0, 25)
        vis.diff_image_heatmap(25, 25)
        vis.skeleton_image_heatmap(10, 10)
        vis.skeleton_image_heatmap(15, 15)
        vis.skeleton_image_heatmap(0, 25)
        vis.skeleton_image_heatmap(25, 25)
        for key in ext.IDs:
            vis.show_skeleton(key)
            vis.conpare_intensity_boxplot(key, func=violinplot)
            vis.conpare_intensity_boxplot(key, func=boxplot)
            vis.conpare_intensity_boxplot(key, func=violinplot, crack_only=False)
            vis.conpare_crack_shape_boxplot(key, func=violinplot)
            print(f"\nKEY {key}: DONE\n")
        print(f"\nENDING {ex_name} ANALYSIS\n")
    print("\nALL DONE")


if __name__ == "__main__":
    start_time = time.time()  # 実行開始時刻を記録
    main()  # 関数の実行
    end_time = time.time()  # 実行終了時刻を記録

    elapsed_time = end_time - start_time  # 経過時間を計算
    minutes = int(elapsed_time // 60)  # 分に変換
    seconds = elapsed_time % 60  # 残りの秒数

    print(f"実行時間: {minutes} 分 {seconds:.2f} 秒")
