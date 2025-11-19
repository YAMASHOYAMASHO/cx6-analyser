from matplotlib import pyplot as plt
import numpy as np
from tkinter import filedialog
from tqdm import tqdm
import math
import os
from matplotlib.colors import ListedColormap, BoundaryNorm
import json
import os
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.widgets import RectangleSelector
import pandas as pd
from pathlib import Path
from util import select_multiple_subdirectories  # 同じパッケージ内の util.py
from scipy.optimize import curve_fit as cf

plt.rcParams["figure.figsize"] = [5, 4]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

def norm(data):
    """
    正規化関数

    Parameters
    ----------
    data : list
        正規化するデータ

    Returns
    -------
    n_data : list
        正規化されたデータ
    """
    max_v = np.max(data)
    min_v = np.min(data)
    n_data = (data - min_v) / (max_v - min_v)
    return n_data


def gauss(x, height, center, sigma, base):
    return base + height * np.exp(-((x - center) ** 2) / (2 * sigma**2))


def gauss_fitting(xx, yy):
    """
    ガウス関数によるフィッティングを行う関数

    Parameters
    ----------
    xx : list
        x軸のデータ
    yy : list
        y軸のデータ

    Returns
    -------
    popt : list
        フィッティングの結果
    """
    x = []
    y = []
    # 0.5以上のデータのみを取り出す
    for i in range(len(yy)):
        if yy[i] >= 0.5:
            x.append(xx[i])
            y.append(yy[i])
    x = np.array(x)
    y = np.array(y)
    # おおよその初期値を設定
    x_max = x[np.argmax(y)]
    # フィッティング
    popt, _ = cf(gauss, x, y, p0=[1, x_max, 0.1, 0])
    return popt

def calc_stress(twothetas, psis, modulus, poisson, wavelength_SPring8):
    """
    応力を計算する関数

    Parameters
    ----------
    twothetas : list
        2θのリスト
    psis : list
        psiのリスト

    Returns
    -------
    result : dict
        応力の計算結果
    """

    # パラメータの読み込み
    E = modulus
    v = poisson
    lam = wavelength_SPring8
    sin2psi = np.sin(np.radians(psis)) ** 2

    # 応力の計算
    d = lam / (2 * np.sin(np.radians(np.array(twothetas) / 2)))
    a1, b1 = np.polyfit(sin2psi, d, 1)
    d0 = b1 + v * a1 / (1 + v)
    e = (d - d0) / d0
    a2, b2 = np.polyfit(sin2psi, e, 1)
    r = a2 * E / (1 + v) * 1000

    # ndarrayをlistに変換
    d = d.tolist()
    e = e.tolist()
    sin2psi = sin2psi.tolist()

    result = {
        "d": d,
        "e": e,
        "r": r,
        "d0": d0,
        "a1": a1,
        "b1": b1,
        "a2": a2,
        "b2": b2,
        "2θ": twothetas,
        "sin2psi": sin2psi,
    }
    return result


# statusファイルを作成する関数
def make_status(
    path,
    name,
    poisson=None,
    modulus=None,
    wavelength_SPring8=None,
    psi_SPring8=None,
    phi_SPring8=None,
    logging=print,
):
    """
    statusファイルを作成する関数

    Parameters
    ----------
    path : str
        フォルダのパス
    name : str
        ファイル名
    poisson : float
        ポアソン比
    modulus : float
        ヤング率
    wavelength_SPring8 : float
        SPring8の波長
    psi_SPring8 : list
        SPring8のpsiのリスト
        psi...手前側に傾ける角度
    phi_SPring8 : list
        SPring8のphiのリスト
        phi...回転する角度
    """
    # ファイル名の作成
    save_path = os.path.join(path, name)
    # フォルダの作成
    Path(path).mkdir(parents=True, exist_ok=True)
    # データの保存
    with open(save_path + ".json", "w") as f:
        json.dump(
            {
                "poisson": poisson,
                "modulus": modulus,
                "wavelength_SPring8": wavelength_SPring8,
                "psi_SPring8": psi_SPring8,
                "name": name,
                "phi_SPring8": phi_SPring8,
            },
            f,
            indent=4,
        )
    return logging("statusファイルを作成しました")


# statusファイルを編集する関数
def edit_status(
    path,
    poisson=None,
    modulus=None,
    wavelength_SPring8=None,
    psi_SPring8=None,
    phi_SPring8=None,
    logging=print,
):
    """
    statusファイルを編集する関数

    Parameters
    ----------
    path : str
        ファイルのパス
    poisson : float
        ポアソン比
    modulus : float
        ヤング率
    wavelength_SPring8 : float
        SPring8の波長
    psi_SPring8 : list
        SPring8のpsiのリスト
    phi_SPring8 : list
        SPring8のphiのリスト
    """
    # データの読み込み
    with open(path + ".json", "r") as f:
        data = json.load(f)
    # データの編集
    if poisson:
        data["poisson"] = poisson
    if modulus:
        data["modulus"] = modulus
    if wavelength_SPring8:
        data["wavelength_SPring8"] = wavelength_SPring8
    if psi_SPring8:
        data["psi_SPring8"] = psi_SPring8
    if phi_SPring8:
        data["phi_SPring8"] = phi_SPring8
    # データの保存
    with open(path + ".json", "w") as f:
        json.dump(data, f, indent=4)
    return (
        logging("statusファイルを編集しました"),
        logging("ポアソン比:", data["poisson"]),
        logging("結晶弾性率:", data["modulus"]),
        logging("SPring8の波長:", data["wavelength_SPring8"]),
        logging("SPring8のψ:", data["psi_SPring8"]),
        logging("SPring8のφ:", data["phi_SPring8"]),
    )

def get_psi_angle(theta, alpha, delta):
    # degreeをradに変換
    theta = np.radians(theta)
    alpha = np.radians(alpha)
    delta = np.radians(delta)

    sin_phi = math.sqrt(1 - (np.cos(delta) * np.cos(theta))**2)
    cos_psi = np.sin(theta) * np.sin(alpha) + sin_phi * np.cos(theta) * np.cos(alpha)
    psi = math.acos(cos_psi)
    return math.degrees(psi)

class SPring8_single:
    def __init__(self, path, time=None, temp=None, alpha=None):
        """
        コンストラクタ 保存済みデータを読み込む際は保存フォルダのパスを指定する

        Parameters
        ----------
        path : str
            保存先フォルダのパス
        """
        self.path = path

        # 元データ
        self.df = pd.read_csv(path + "/.csv")
        self.xmin = self.df["2θ(deg)"].min()
        self.xmax = self.df["2θ(deg)"].max()
        if '.json' in os.listdir(path):
            # パスが指定されている場合はデータを読み込む
            with open(path + "/.json", "r") as f:
                data = json.load(f)
            # パラメータ
            self.poisson = data["poisson"]
            self.modulus = data["modulus"]
            self.wavelength_SPring8 = data["wavelength_SPring8"]
            self.psi_SPring8 = data["psi_SPring8"]
            self.alpha = data.get("alpha")  # 入射角
            self.delta = data.get("delta")  # 方位角
            self.is_paras = data["is_paras"]
            self.time = data["time"]
            self.temp = data['temp']
            # フィッティング結果
            self.paras: list = data["paras"] if isinstance(data["paras"], list) else np.array(data["paras"]).tolist()  # type: ignore
            self.twothetas: list = data["twothetas"] if isinstance(data["twothetas"], list) else np.array(data["twothetas"]).tolist()  # type: ignore
            self.is_fitting = data["is_fitting"]
            # 応力の計算結果
            self.a1 = data["a1"]
            self.b1 = data["b1"]
            self.a2 = data["a2"]
            self.b2 = data["b2"]
            self.d0 = data["d0"]
            self.d = data["d"] if isinstance(data["d"], list) else np.array(data["d"]).tolist()  # type: ignore
            self.e = data["e"] if isinstance(data["e"], list) else np.array(data["e"]).tolist()  # type: ignore
            self.r = data["r"]
            self.sin2psi = data["sin2psi"] if isinstance(data["sin2psi"], list) else np.array(data["sin2psi"]).tolist()  # type: ignore
            self.is_calculated = data["is_calculated"]
            self.bools = data["bools"]
            self.ci_r = data["ci_r"]
            # 名前
            self.name = data["name"]
            self.is_saved = data["is_saved"]

        else:
            for column in self.df.columns[1:]:
                self.df[column] = norm(self.df[column])
            if None in [time, temp, alpha]:
                raise ValueError("time, temp, alpha が指定されていません。新規作成する場合は適切な処理を実装してください。")
            # 初期化
            # パラメータ
            self.poisson = None
            self.modulus = None
            self.wavelength_SPring8 = None
            self.psi_SPring8 = []
            self.alpha = alpha  # 入射角
            self.delta = []
            self.is_paras = False
            self.time = time
            self.temp = temp
            # フィッティング結果
            self.paras: list = []
            self.twothetas: list = []
            self.is_fitting = False
            # 応力の計算結果
            self.a1 = None
            self.b1 = None
            self.a2 = None
            self.b2 = None
            self.d0 = None
            self.d = []
            self.e = []
            self.r = None
            self.sin2psi = []
            self.is_calculated = False
            self.bools = []
            self.ci_r = None
            # 名前
            self.name = None
            self.is_saved = False

    def set_status(self, status, logging=print):
        """
        各物性値を設定する関数

        Parameters
        ----------
        status : dict
            物性値を格納した辞書 例: {"poisson": 0.3, "modulus": 100, "wavelength_SPring8": 0.154, "psi_SPring8": [0, 45, 90]}
        logging : function
            ログを出力する関数 デフォルトはprint
        """
        self.poisson = status["poisson"]
        self.modulus = status["modulus"]
        self.wavelength_SPring8 = status["wavelength_SPring8"]
        self.is_paras = True

    def fitting(self, coding=True, logging=print, result_collector=None):
        """
        フィッティングを行う関数 アプリに組み込む場合，coding=Falseで確認を無効にする

        Parameters
        ----------
        coding : bool
            coding内での使用を想定しているか
        logging : function
            ログを出力する関数 デフォルトはprint
        result_collector : ProcessingResult, optional
            処理結果を収集するオブジェクト
        """
        try:
            # データの有無の確認
            if self.is_fitting:  # フィッティング済みの場合は初期化
                if coding:
                    ans = input("フィッティング済みです 再フィッティングしますか？(y/n)")
                    if ans == "n":
                        return logging("キャンセルしました")
                self.paras = []
                self.twothetas = []

            # フィッティング
            miss_fitting = 0
            for i in range(1, len(self.df.columns)):
                try:
                    para = gauss_fitting(self.df["2θ(deg)"], self.df[self.df.columns[i]])
                    self.paras.append(para)
                    delta = float(self.df.columns[i])
                    self.delta.append(delta)
                    psi = get_psi_angle(para[1], self.alpha, delta)
                    self.psi_SPring8.append(psi)
                except Exception as e:
                    logging(f"フィッティングに失敗しました: パス={self.path}, delta={self.df.columns[i]}")
                    logging(e)
                    miss_fitting += 1
            self.paras = np.array(self.paras)  # type: ignore
            self.twothetas = self.paras[:, 1].tolist()  # type: ignore
            self.is_fitting = True
            
            if miss_fitting == len(self.df.columns) - 1:
                logging("全てのフィッティングに失敗しました")
                return
            if miss_fitting > len(self.df.columns) - 4:
                logging("フィッティングに多数失敗しました")
                return

            logging("フィッティングが完了しました")
            return self.calc_stress(logging=logging)
            
        except Exception as e:
            if result_collector:
                result_collector.add_error("fitting", e)
            raise

    def calc_stress(self, logging=print):
        """
        応力を計算する関数

        Parameters
        ----------
        logging : function
            ログを出力する関数 デフォルトはprint
        """
        # パラメータの確認
        if not self.is_paras:
            raise ValueError("パラメータが設定されていません")
        if self.wavelength_SPring8 is None:
            raise ValueError("wavelength_SPring8が設定されていません")
        if self.poisson is None:
            raise ValueError("poissonが設定されていません")
        if self.modulus is None:
            raise ValueError("modulusが設定されていません")

        # d0の計算
        lam = self.wavelength_SPring8 * 1e-10  # 波長 (m)
        v = self.poisson  # ポアソン比
        sin2psi = np.sin(np.radians(self.psi_SPring8)) ** 2
        d = lam / (
            2 * np.sin(np.radians(np.array(self.twothetas) / 2))
        )  # Braggの式からdを計算 (m)
        a1, b1 = np.polyfit(sin2psi, d, 1)  # sin2ψ vs. d の直線のフィッティング
        d0 = b1 + v * a1 / (1 + v)  # sin2ψ=v/(1+v)のときにd0になる (m)
        sintheta0 = lam / (2 * d0)  # sinθ0の計算
        theta0 = np.arcsin(sintheta0)  # θ0の計算

        # 応力の計算
        K = (
            -self.modulus / (2 * (1 + self.poisson)) * np.pi / 180 / np.tan(theta0)
        )  # (MPa/deg.)
        M, A = np.polyfit(sin2psi, self.twothetas, 1)  # M (deg.)
        r = K * M  # (GPa)
        r = r * 1e3  # (MPa)

        # 信頼区間の計算
        n = len(self.psi_SPring8)
        t_distribution = {
            1: 1.839,
            2: 1.321,
            3: 1.197,
            4: 1.142,
            5: 1.110,
            6: 1.090,
            7: 1.077,
            8: 1.067,
            9: 1.059,
            10: 1.053,
            11: 1.048,
            12: 1.044,
            13: 1.041,
            14: 1.038,
            15: 1.035,
            16: 1.033,
        }  # t分布のt値    ただし，自由度(n-2), 信頼率(1-α) = 0.683のとき
        if n < 3:  # データ数が足りない場合
            raise ValueError("データ数が足りません")
        t = t_distribution[n - 2]  # t値
        y_hat = M * sin2psi + A  # 予測値
        s = self.twothetas - y_hat  # 残差
        s2 = np.sum(s**2) / (n - 2)  # 残差分散
        se = np.sqrt(s2 / np.sum((sin2psi - np.mean(sin2psi)) ** 2))
        ci = t * se  # 信頼区間係数
        ci_r = ci * K  # 応力の信頼区間
        ci_r = abs(ci * K)  # 応力の信頼区間の絶対値 (GPa)
        ci_r = ci_r * 1e3  # 応力の信頼区間の絶対値 (MPa)

        # e vs. sin2ψ の直線の傾きと切片
        e = d / d0 - 1
        a2, b2 = np.polyfit(sin2psi, e, 1)

        # 結果の格納
        self.a1 = a1
        self.b1 = b1
        self.a2 = a2
        self.b2 = b2
        self.d0 = d0
        self.d = d
        self.e = e
        self.r = r
        self.sin2psi = sin2psi
        self.is_calculated = True
        self.ci_r = ci_r
        return logging("応力の計算が完了しました")

    def recalc_stress(self, bools, logging=print):
        """
        特定のψのデータを除き，再計算を行う関数

        Parameters
        ----------
        bools : list
            どのデータを使用して再計算するかをboolで指定
        logging : function
            ログを出力する関数 デフォルトはprint
        """
        # パラメータの確認
        if not self.is_paras:
            raise ValueError("パラメータが設定されていません")

        # フィッティングの確認
        if not self.is_fitting:
            raise ValueError("フィッティングが行われていません")

        # boolsの確認
        if len(bools) != len(self.psi_SPring8):
            raise ValueError("boolsの長さがpsi_SPring8の長さと一致しません")

        # 応力の再計算
        thothetas = []
        psis = []
        for i, bool in enumerate(bools):
            if bool:
                thothetas.append(self.twothetas[i])
                psis.append(self.psi_SPring8[i])
        result = calc_stress(
            thothetas,
            psis,
            self.modulus,
            self.poisson,
            self.wavelength_SPring8,
        )
        self.a1 = result["a1"]
        self.b1 = result["b1"]
        self.a2 = result["a2"]
        self.b2 = result["b2"]
        self.d0 = result["d0"]
        self.d = result["d"]
        self.e = result["e"]
        self.r = result["r"]
        self.sin2psi = result["sin2psi"]
        self.is_calculated = True
        self.bools = bools
        return logging("応力の再計算が完了しました")

    def save_datas(self, name, logging=print, result_collector=None):
        """
        データを保存する関数

        Parameters
        ----------
        path : str
            保存先のフォルダのパス
        name : str
            保存フォルダ名
        logging : function
            ログを出力する関数 デフォルトはprint
        result_collector : ProcessingResult, optional
            処理結果を収集するオブジェクト
        """
        try:
            self.name = name
            # データの保存
            with open(self.path + "/.json", "w") as f:
                json.dump(
                    {
                        "poisson": self.poisson,
                        "modulus": self.modulus,
                        "wavelength_SPring8": self.wavelength_SPring8,
                        "psi_SPring8": self.psi_SPring8,
                        "alpha": self.alpha,
                        "delta": self.delta,
                        "time": self.time, 
                        "temp": self.temp,
                        "paras": self.paras.tolist() if isinstance(self.paras, np.ndarray) else self.paras,
                        "twothetas": self.twothetas if isinstance(self.twothetas, list) else self.twothetas.tolist(),
                        "a1": self.a1,
                        "b1": self.b1,
                        "a2": self.a2,
                        "b2": self.b2,
                        "d0": self.d0,
                        "d": self.d.tolist() if isinstance(self.d, np.ndarray) else self.d,
                        "e": self.e.tolist() if isinstance(self.e, np.ndarray) else self.e,
                        "r": self.r,
                        "sin2psi": self.sin2psi.tolist() if isinstance(self.sin2psi, np.ndarray) else self.sin2psi,
                        "name": self.name,
                        "ci_r": self.ci_r,
                        "bools": self.bools,
                        "is_saved": self.is_saved,
                        "is_calculated": self.is_calculated,
                        "is_fitting": self.is_fitting,
                        "is_paras": self.is_paras,
                        "path": self.path,
                        "xmin": self.xmin,
                        "xmax": self.xmax,
                    },
                    f,
                    indent=4,
                )
            self.df.to_csv(self.path + "/.csv", index=False)
            self.is_saved = True
            
            if result_collector:
                result_collector.add_success(name, stress=self.r, ci_stress=self.ci_r, details="データ保存完了")
            
            return logging("データを保存しました")
            
        except Exception as e:
            if result_collector:
                result_collector.add_error(name, e)
            raise

    # coding専用関数
    def plot_raw_data(self, save_path=None, logging=print):
        """
        生データをプロットする関数
        coding専用関数

        Parameters
        ----------
        save_path : str
            保存先のパス デフォルトはNone(保存しない)
        logging : function
            ログを出力する関数 デフォルトはprint
        """
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["svg.fonttype"] = "none"
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        fig, ax = plt.subplots(figsize=(5, 4))

        # パラメータの確認
        if self.is_paras:
            label = [f"ψ={psi}" for psi in self.psi_SPring8]
        else:
            label = [f"real_{i}" for i in range(1, len(self.df.columns))]

        # プロット
        cmap = cm.get_cmap('GnBu')
        for i in range(1, len(self.df.columns)):
            ax.plot(
                self.df["2θ(deg)"],
                self.df[f"real_{i}"] + (i - 1) * 0.2,
                label=label[i - 1],
                color=cmap((i + 4) / (len(self.df.columns) + 4)),
            )
        # プロットの詳細設定
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_xticks(np.arange(self.xmin, self.xmax + 0.1, 0.2))
        ax.set_yticks([])
        ax.set_xlabel("2θ(deg)")
        ax.set_ylabel("Intensity")
        ax.legend()
        plt.tight_layout()
        if save_path:
            # 拡張子を.svgに強制
            base, ext = os.path.splitext(save_path)
            save_path_svg = base + ".svg"
            plt.savefig(save_path_svg)
            print(f"プロットを {save_path_svg} に保存しました")

    def plot_fitting(self, save_path=None, logging=print):
        """
        フィッティング結果をプロットする関数
        coding専用関数

        Parameters
        ----------
        save_path : str
            保存先のパス デフォルトはNone(保存しない)
        logging : function
            ログを出力する関数 デフォルトはprint
        """
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["svg.fonttype"] = "none"
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        fig, ax = plt.subplots(figsize=(5, 4))
        # データの有無の確認
        if not self.is_fitting:
            raise ValueError("フィッティングが行われていません")

        # パラメータの確認
        if self.is_paras:
            label = [f"ψ={psi}" for psi in self.psi_SPring8]
        else:
            label = [f"real_{i}" for i in range(1, len(self.df.columns))]

        # プロット
        cmap_blue = cm.get_cmap('GnBu')
        cmap_orange = cm.get_cmap('Oranges')
        for i in range(1, len(self.df.columns)):  # 生データ
            ax.plot(
                self.df["2θ(deg)"],
                self.df[f"real_{i}"] + (i - 1) * 0.2,
                label=label[i - 1],
                color=cmap_blue((i + 4) / (len(self.df.columns) + 4)),
            )
        for i, para in enumerate(self.paras):  # フィッティング
            ax.plot(
                self.df["2θ(deg)"],
                gauss(self.df["2θ(deg)"], *para) + i * 0.2,
                color=cmap_orange((i + 4) / (len(self.df.columns) + 4)),
            )
        for i, twotheta in enumerate(self.twothetas):  # ピーク
            ax.plot(twotheta, 0.9 + i * 0.2, "^k")
        # プロットの詳細設定
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_xticks(np.arange(self.xmin, self.xmax + 0.1, 0.2))
        ax.set_yticks([])
        ax.set_xlabel("2θ(deg)")
        ax.set_ylabel("Intensity")
        ax.legend()
        plt.tight_layout()
        if save_path:
            # 拡張子を.svgに強制
            base, ext = os.path.splitext(save_path)
            save_path_svg = base + ".svg"
            plt.savefig(save_path_svg)
            print(f"プロットを {save_path_svg} に保存しました")

    def plot_d(self, save_path=None, logging=print):
        """
        d vs. sin2ψ のプロットを行う関数
        coding専用関数

        Parameters
        ----------
        save_path : str
            保存先のパス デフォルトはNone(保存しない)
        logging : function
            ログを出力する関数 デフォルトはprint
        """
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["svg.fonttype"] = "none"
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        fig, ax = plt.subplots(figsize=(5, 4))
        # データの有無の確認
        if not self.is_calculated:
            raise ValueError("計算が行われていません")

        # プロット
        x = np.array([0, 0.8])
        d = self.a1 * x + self.b1
        ax.plot(x, d, "--")
        ax.plot(self.sin2psi, self.d, "o")
        # プロットの詳細設定
        ax.set_xlim(0, 0.8)
        ax.set_xlabel(r"sin$^2\psi$")
        ax.set_ylabel("d [nm]")
        plt.tight_layout()
        if save_path:
            # 拡張子を.svgに強制
            base, ext = os.path.splitext(save_path)
            save_path_svg = base + ".svg"
            plt.savefig(save_path_svg)
            print(f"プロットを {save_path_svg} に保存しました")

    def plot_e(self, save_path=None, logging=print):
        """
        ε vs. sin2ψ のプロットを行う関数
        coding専用関数

        Parameters
        ----------
        save_path : str
            保存先のパス デフォルトはNone(保存しない)
        logging : function
            ログを出力する関数 デフォルトはprint
        """
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["svg.fonttype"] = "none"
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        fig, ax = plt.subplots(figsize=(5, 4))
        # データの有無の確認
        if not self.is_calculated:
            raise ValueError("計算が行われていません")

        # プロット
        x = np.array([0, 0.8])
        e = self.a2 * x + self.b2
        ax.plot(x, e, "--")
        ax.plot(self.sin2psi, self.e, "o")
        # プロットの詳細設定
        ax.set_xlim(0, 0.8)
        ax.set_ylim(-0.003, 0.003)
        ax.set_xlabel(r"sin$^2\psi$")
        ax.set_ylabel("ε")
        plt.tight_layout()
        if save_path:
            # 拡張子を.svgに強制
            base, ext = os.path.splitext(save_path)
            save_path_svg = base + ".svg"
            plt.savefig(save_path_svg)
            print(f"プロットを {save_path_svg} に保存しました")