from calendar import c
from matplotlib import pyplot as plt
import numpy as np
import fabio
from tkinter import filedialog
from tqdm import tqdm
import math
import pyFAI.detectors
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from traitlets import default
import os
from matplotlib.colors import ListedColormap, BoundaryNorm
import json
import os
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.widgets import RectangleSelector
import pandas as pd
from pathlib import Path
from util import select_deep_files  # 同じパッケージ内の util.py

plt.rcParams["figure.figsize"] = [10, 8]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

# 入力項目について
ALPHA = 2  # degree
TWOTHETA = 12.2  # degree
WAVELENGTH = 0.1 * 1e-9  # m
REF_DISTANCE = 0.6161500 * 1e-9  # m   標準試料
PIXEL_SIZE = 172 * 1e-6  # m
DEFAULT_DIR = '//nishinolab-data/Public/SPring-8'
AZIMUTH_ANGLES = [90, 80, 70, 60, 50, 40, 30, 20, 10]
RADIAL_INIT = 11.8
RADIAL_END = 12.8
AZIMUTH_WIDTH = 2.0  # degree
AZIMUTH_ABS = 1  
SAVE_PATH = str((Path(__file__).resolve().parents[3] / "datas" / "SPring8").resolve())
QUADLAND = 1  # 象限設定 (0: 右半分, 1: 左半分)

# Configの設定
class Config:
    alpha = ALPHA
    twoTheta = TWOTHETA
    wavelength = WAVELENGTH
    ref_distance = REF_DISTANCE
    pixel_size = PIXEL_SIZE
    camera_length = 0.0
    beam_center = (0.0, 0.0)
    azimuth_angles = AZIMUTH_ANGLES
    radial_init = RADIAL_INIT
    radial_end = RADIAL_END
    azimuth_width = AZIMUTH_WIDTH
    azimuth_abs = AZIMUTH_ABS
    quadland = QUADLAND

# 画像の選択
REF_PATH = filedialog.askopenfilename(
    title="標準試料を選択してください",
    initialdir=DEFAULT_DIR,
    filetypes=[("TIF files", "*.tif"), ("All files", "*.*")],
)
AIR_PATH = filedialog.askopenfilename(
    title="air画像を選択してください",
    initialdir=DEFAULT_DIR,
    filetypes=[("TIF files", "*.tif"), ("All files", "*.*")],
)


class Path:
    default_dir = DEFAULT_DIR
    ref = REF_PATH
    air = AIR_PATH
    data = []
    save = SAVE_PATH


# 画像の読み込み
REF_IMAGE_ORIGIN = fabio.open(Path.ref).data
AIR_IMAGE = fabio.open(Path.air).data
REF_IMAGE = REF_IMAGE_ORIGIN - AIR_IMAGE


class Image:
    ref_origin = REF_IMAGE_ORIGIN
    air = AIR_IMAGE
    ref = REF_IMAGE
    mask = None
    example = None


# 画像の確認
_vmax = 300
while _vmax:
    vmax = _vmax
    plt.imshow(Image.ref, vmax=vmax, cmap="gray", origin="lower")
    plt.colorbar()
    plt.title(f"Reference Image (vmax={vmax})")
    plt.show()
    _vmax = int(input("次のvmaxを入力してください (終了する場合は0を入力): "))
print("vmaxの設定が完了しました。")

# STEP1: リファレンス画像の中心位置の決定
# ピーク位置の自動検出
_upper = 5  # %
while _upper:
    upper = _upper
    peak_position = np.where(Image.ref > np.percentile(Image.ref, 100 - upper))
    plt.imshow(Image.ref, vmax=vmax, cmap="gray", origin="lower")
    plt.scatter(peak_position[1], peak_position[0], s=1, c="red")
    plt.title(f"Peak Positions (upper {upper}%)")
    plt.show()
    _upper = float(input("次のupper%を入力してください (終了する場合は0を入力): "))
print("ピーク位置の設定が完了しました。")

# ピーク位置の選択
is_peak = 1
while is_peak:
    positions = []

    fig, ax = plt.subplots()
    ax.imshow(Image.ref, vmax=vmax, cmap="gray", origin="lower")
    point_plot_init = ax.plot(peak_position[1], peak_position[0], "ro", markersize=1)[0]
    points_plot = ax.plot([], [], "go", markersize=1)[0]
    plt.title("ピーク位置の選択 (左クリック: 追加, 右クリック: 除外)")

    selection_mode = {"add": True}  # True: 追加モード, False: 除外モード

    def onselect(eclick, erelease):
        global positions
        x0 = min(eclick.xdata, erelease.xdata)
        x1 = max(eclick.xdata, erelease.xdata)
        y0 = min(eclick.ydata, erelease.ydata)
        y1 = max(eclick.ydata, erelease.ydata)

        # 選択範囲内の点を抽出
        selected = [
            (p[1], p[0])
            for p in zip(peak_position[0], peak_position[1])
            if x0 <= p[1] <= x1 and y0 <= p[0] <= y1
        ]
        if eclick.button == 1:  # 左クリック: 追加
            positions = list(set(positions + selected))
            print(f"追加: {len(selected)}点 (合計: {len(positions)}点)")
        elif eclick.button == 3:  # 右クリック: 除外
            positions = [p for p in positions if p not in selected]
            print(f"除外: {len(selected)}点 (残り: {len(positions)}点)")

        # プロット更新
        points_plot.set_data([x[0] for x in positions], [x[1] for x in positions])
        fig.canvas.draw_idle()

    rect_selector = RectangleSelector(
        ax,
        onselect,
        useblit=True,
        button=[1, 3],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True,
    )
    plt.show()
    is_peak = int(input("ピーク選択を続けますか？ (続ける:1, 終了:0): "))
print("ピーク位置の選択が完了しました。")


# 円のフィッティング関数
def CircleFitting(x, y):
    """最小二乗法による円フィッティングをする関数
    input: x,y 円フィッティングする点群

    output  cxe 中心x座標
            cye 中心y座標
            re  半径

    参考
    一般式による最小二乗法（円の最小二乗法）　画像処理ソリューション
    http://imagingsolution.blog107.fc2.com/blog-entry-16.html
    """

    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum([ix**2 for ix in x])
    sumy2 = sum([iy**2 for iy in y])
    sumxy = sum([ix * iy for (ix, iy) in zip(x, y)])

    F = np.array([[sumx2, sumxy, sumx], [sumxy, sumy2, sumy], [sumx, sumy, len(x)]])

    G = np.array(
        [
            [-sum([ix**3 + ix * iy**2 for (ix, iy) in zip(x, y)])],
            [-sum([ix**2 * iy + iy**3 for (ix, iy) in zip(x, y)])],
            [-sum([ix**2 + iy**2 for (ix, iy) in zip(x, y)])],
        ]
    )

    T = np.linalg.inv(F).dot(G)

    cxe = float(T[0] / -2)
    cye = float(T[1] / -2)
    re = math.sqrt(cxe**2 + cye**2 - T[2])
    # print (cxe,cye,re)
    return (cxe, cye, re)


# 円フィッティング
x = [x[0] for x in positions]
y = [x[1] for x in positions]
cxe, cye, re = CircleFitting(x, y)
Config.beam_center = (cxe, cye)
# print(cxe, cye, re)

# 結果の表示
plt.imshow(Image.ref, cmap="gray", vmax=vmax, origin="lower")
plt.plot([x[0] for x in positions], [x[1] for x in positions], "ro")
plt.plot(cxe, cye, "bo")
circle = plt.Circle((cxe, cye), re, color="b", fill=False)
plt.gca().add_artist(circle)
plt.show()

# カメラ長算出のための係数
theta = math.asin(Config.wavelength / (2 * Config.ref_distance))  # rad
theta = math.degrees(theta)  # degree

# camera lengthの計算
radius = re
camera_length = radius * Config.pixel_size / math.tan(math.radians(2 * theta))  # m
print(f"Calculated camera length: {camera_length*1e2:.2f} cm")
Config.camera_length = camera_length

# STEP2: マスクの作成
mask = np.zeros_like(Image.air)
mask[Image.air < 0] = 1
plt.imshow(mask, cmap="gray", origin="lower")
plt.title("Mask Image")
plt.show()
is_mask = int(input("マスクを作成しますか？ (作成する:1, 作成しない:0): "))
if is_mask:
    Image.mask = mask

# STEP3: AzimuthalIntegratorの設定
detector = pyFAI.detectors.Detector(
    pixel1=Config.pixel_size, pixel2=Config.pixel_size, max_shape=Image.ref.shape
)
ai = AzimuthalIntegrator(
    dist=Config.camera_length, detector=detector, wavelength=Config.wavelength
)
ai.setFit2D(Config.camera_length*1000, Config.beam_center[0], Config.beam_center[1])
print("AzimuthalIntegratorの設定が完了しました。")

# STEP4: 1D変換の実行
# 解析データのパスを取得
# ★★★今後変更の可能性あり★★★->外部モジュールで管理
is_path = 0
while not is_path:
    DATA_PATHS = select_deep_files()
    if DATA_PATHS:
        is_path = 1
    else:
        print("ファイルが選択されませんでした。再度選択してください。")
print("解析対象ファイル数:", len(DATA_PATHS))
Path.data = DATA_PATHS
EXAMPLE_IMAGE = fabio.open(Path.data[0]).data - Image.air
EXAMPLE_IMAGE = EXAMPLE_IMAGE - EXAMPLE_IMAGE.min() + 1
EXAMPLE_IMAGE[Image.mask == 1] = 0
Image.example = EXAMPLE_IMAGE

# revmaxの設定
_revmax = 2000
while _revmax:
    revmax = _revmax
    plt.imshow(Image.example, vmax=revmax, cmap="gray", origin="lower")
    plt.colorbar()
    plt.title(f"Example Data Image (vmax={revmax})")
    plt.show()
    _revmax = int(input("次のvmaxを入力してください (終了する場合は0を入力): "))
print("vmaxの設定が完了しました。")

# azimuth角の確認
colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(Config.azimuth_angles)))
is_edit = 1
while is_edit:
    plt.imshow(Image.example, vmax=revmax, cmap="gray", origin="lower")
    for i, angle in enumerate(Config.azimuth_angles):
        # azimuth角のプロット
        x_vals = np.array([0, Image.example.shape[1]])
        y_vals_up = Config.beam_center[1] + np.tan(np.radians(angle * Config.azimuth_abs + 2)) * (x_vals - Config.beam_center[0])
        y_vals_down = Config.beam_center[1] + np.tan(np.radians(angle * Config.azimuth_abs - 2)) * (x_vals - Config.beam_center[0])
        plt.plot(x_vals, y_vals_up, color=colors[i])
        plt.plot(x_vals, y_vals_down, color=colors[i])
        plt.fill_between(x_vals, y_vals_up, y_vals_down, color=colors[i], alpha=0.3)

    # 中心位置からの距離を計算
    def calculate_distance_from_center(camera_length, two_theta, pixel_size):
        return camera_length * np.tan(np.radians(two_theta)) / pixel_size

    inner = calculate_distance_from_center(Config.camera_length, Config.radial_init, Config.pixel_size)
    outer = calculate_distance_from_center(Config.camera_length, Config.radial_end, Config.pixel_size)
    circle_inner = plt.Circle((cxe, cye), inner, color='b', fill=False)
    circle_outer = plt.Circle((cxe, cye), outer, color='b', fill=False)
    plt.gca().add_artist(circle_inner)
    plt.gca().add_artist(circle_outer)
        
    xlim = Image.example.shape[1]
    ylim = Image.example.shape[0]
    plt.xlim(0, xlim)
    plt.ylim(0, ylim)
    plt.show()
    is_edit = int(input("角度の編集を続けますか？ (続ける:1, 終了:0): "))
    if is_edit:
        try:
            is_edit = int(input("どの角度を編集しますか？ (azimuth角:1, twotheta範囲:2, azimuth幅:3. azimuth絶対値:4, 象限:5): "))
            if is_edit == 1:
                new_azimuth = input("新しいazimuth角をカンマ区切りで入力してください (例: 90,80,70): ")
                Config.azimuth_angles = [float(d.strip()) for d in new_azimuth.split(",")]
            elif is_edit == 2:
                new_init = float(input("新しいradial_initを入力してください (現在値: {}): ".format(Config.radial_init)))
                new_end = float(input("新しいradial_endを入力してください (現在値: {}): ".format(Config.radial_end)))
                Config.radial_init = new_init
                Config.radial_end = new_end
            elif is_edit == 3:
                new_width = float(input("新しいazimuth幅を入力してください (現在値: {}): ".format(Config.azimuth_width)))
                Config.azimuth_width = new_width
            elif is_edit == 4:
                Config.azimuth_abs = Config.azimuth_abs * -1
            elif is_edit == 5:
                new_quadland = int(input("新しい象限を入力してください (0: 右半分, 1: 左半分) (現在値: {}): ".format(Config.quadland)))
                Config.quadland = new_quadland
        except:
            print("入力に誤りがあります。再度入力してください。")
print("azimuth角とtwotheta範囲の設定が完了しました。")

# STEP5: 1D変換の実行
for path in tqdm(Path.data, desc="1D変換中"):
    path = path.replace("\\", "/")
    # 親保存フォルダの作成
    file_name = os.path.splitext(os.path.basename(path))[0][:-6]    # {file_name}_00001.tif の形式を想定
    save_path_parent = os.path.join(Path.save, file_name)
    os.makedirs(save_path_parent, exist_ok=True)
    # Configの保存
    config_dict = {
        "alpha": Config.alpha,
        "twoTheta": Config.twoTheta,
        "wavelength": Config.wavelength,
        "ref_distance": Config.ref_distance,
        "pixel_size": Config.pixel_size,
        "camera_length": Config.camera_length,
        "beam_center": list(Config.beam_center),  # tupleをlistに変換
        "azimuth_angles": Config.azimuth_angles,
        "radial_init": Config.radial_init,
        "radial_end": Config.radial_end,
        "azimuth_width": Config.azimuth_width,
        "azimuth_abs": Config.azimuth_abs,
        "quadland": Config.quadland,
        "ref_path": Path.ref,
        "air_path": Path.air,
    }
    if not os.path.exists(os.path.join(save_path_parent, "config.json")):
        with open(os.path.join(save_path_parent, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=4)
    # 保存フォルダの作成
    save_path = os.path.join(save_path_parent, os.path.splitext(os.path.basename(path))[0][-5:])  # {file_name}/00001/
    os.makedirs(save_path, exist_ok=True)
    try:
        img = fabio.open(path).data
        # airを引く
        img -= Image.air
        img = img - img.min() + 1  # 負の値を0に補正
        # integrate

        for i, angle in enumerate(Config.azimuth_angles):            
            res = ai.integrate1d(img, 1300, 
                                    unit="2th_deg",
                                    radial_range=(Config.radial_init, Config.radial_end),
                                    azimuth_range=(
                                        angle*Config.azimuth_abs-Config.azimuth_width+180*(Config.quadland),
                                        angle*Config.azimuth_abs+Config.azimuth_width+180*(Config.quadland)
                                    ),
                                    mask=Image.mask
                            )
            if i == 0:
                df = pd.DataFrame({
                    "2θ(deg)": res.radial,
                    f"{angle}": res.intensity
                })
            else:
                df[f'{angle}'] = res.intensity
        save_csv_path = os.path.join(save_path, ".csv")
        df.to_csv(save_csv_path, index=False)
            
    except Exception as e:
        print(f"1D変換中にエラーが発生しました: {e}")
        continue

print("1D変換が完了しました。")