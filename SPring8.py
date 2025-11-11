import json
import pandas as pd
from SmartLab import gauss_fitting, norm, gauss
import os
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


class SP8_data:
    """
    SPring-8のデータを管理するクラス

    Attributes
    ----------
    path : str
        元データのtxtファイルバス
    poisson : float
        ポアソン比
    modulus : float
        結晶弾性率
    wavelength_SPring8 : float
        SPring8の波長 基本は1 A = 0.1 nm
    psi_SPring8 : list
        SPring8のψのリスト
    is_paras : bool
        各物性値が設定されているか
    df : pd.DataFrame
        元データを格納したDataFrame 2θと各データセットの値
    xmin : float
        2θの最小値
    xmax : float
        2θの最大値
    is_data : bool
        元データが読み込まれているか
    paras : np.array
        フィッティングの結果 リストの各項目にはガウス関数のパラメータが格納されている [height, center, sigma, base]
    twothetas : np.array
        各ψにおけるピークの2θのリスト
    is_fitting : bool
        フィッティングが行われているか
    a1 : float
        sin2ψ vs. d の直線の傾き
    b1 : float
        sin2ψ vs. d の直線の切片
    a2 : float
        sin2ψ vs. ε の直線の傾き
    b2 : float
        sin2ψ vs. ε の直線の切片
    d0 : float
        計算によって得られた無ひずみ結晶面間隔
    d : list
        各ψにおける結晶面間隔dのリスト
    e : list
        各ψにおける結晶ひずみεのリスト
    r : float
        応力の計算結果
    ci_r : float
        応力の信頼区間
    sin2psi : list
        sin2ψのリスト
    is_calculated : bool
        応力の計算が行われているか
    bools : list
        測定された各ψにおけるデータを応力計算時に使用するかのリスト
    name : str
        サンプル名 この名前がそのままファイル名になる
    is_saved : bool
        データが保存されているか
    kind : str
        SPring8_datasでデータをまとめる際に使用される，このデータが単一サンプルのものか複数の平均データかを示す
    temp : float
        測定時の温度 (K)

    Methods
    -------
    set_status(status, logging=print)
        各物性値をセットする関数
    read_txt(data_path, logging=print)
        元データを読み込む関数
    fitting(coding=True, logging=print)
        フィッティングを行う関数
    recalc_stress(bools, logging=print)
        応力の再計算を行う関数
    save_datas(path, name, logging=print)
        データを保存する関数
    plot_raw_data(save_path=None, logging=print)
        生データをプロットする関数
    plot_fitting(save_path=None, logging=print)
        フィッティング結果をプロットする関数
    plot_d(save_path=None, logging=print)
        sin2ψ vs. d のプロットを行う関数
    plot_e(save_path=None, logging=print)
        sin2ψ vs. ε のプロットを行う関数

    See Also
    --------
    SPring8_datas : SPring8の複数データの平均値等を管理するクラス
    SPring8360_data : SPring8での360°3軸応力測定のデータを管理するクラス

    Examples
    --------
    >>> data = SPring8_data()
    >>> data.set_status({"poisson": 0.3, "modulus": 100, "wavelength_SPring8": 0.154, "psi_SPring8": [0, 45, 90]})
    >>> data.read_txt("data.txt")
    >>> data.recalc_stress([True, False, True, True, True, True])
    """

    def __init__(self, path=None):
        """
        コンストラクタ 保存済みデータを読み込む際は保存フォルダのパスを指定する

        Parameters
        ----------
        path : str
            保存先フォルダのパス
        """
        self.path = path
        if path:
            # パスが指定されている場合はデータを読み込む
            with open(path + "/.json", "r") as f:
                data = json.load(f)
            # パラメータ
            self.poisson = data["poisson"]
            self.modulus = data["modulus"]
            self.wavelength_SPring8 = data["wavelength_SPring8"]
            self.psi_SPring8 = data["psi_SPring8"]
            self.is_paras = data["is_paras"]
            self.temp = data.get("temp")  # temp を読み込む (存在しない場合は None)
            # 元データ
            self.df = pd.read_csv(path + "/.csv")
            self.xmin = self.df["2θ"].min()
            self.xmax = self.df["2θ"].max()
            self.is_data = data["is_data"]
            # フィッティング結果
            self.paras = np.array(data["paras"])
            self.twothetas = np.array(data["twothetas"])
            self.is_fitting = data["is_fitting"]
            # 応力の計算結果
            self.a1 = data["a1"]
            self.b1 = data["b1"]
            self.a2 = data["a2"]
            self.b2 = data["b2"]
            self.d0 = data["d0"]
            self.d = data["d"]
            self.e = data["e"]
            self.r = data["r"]
            self.sin2psi = data["sin2psi"]
            self.is_calculated = data["is_calculated"]
            self.bools = data["bools"]
            self.ci_r = data["ci_r"]
            # 名前
            self.name = data["name"]
            self.is_saved = data["is_saved"]

        else:
            # パスが指定されていない場合は初期化
            # パラメータ
            self.poisson = None
            self.modulus = None
            self.wavelength_SPring8 = None
            self.psi_SPring8 = []
            self.is_paras = False
            self.temp = None  # temp を初期化
            # 元データ
            self.df = None
            self.xmin = None
            self.xmax = None
            self.is_data = False
            # フィッティング結果
            self.paras = np.array([])
            self.twothetas = np.array([])
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

        # クラスの種類
        self.kind = "single"

    def set_status(self, status, logging=print):
        """
        各物性値を設定する関数

        Parameters
        ----------
        status : dict
            物性値を格納した辞書 例: {"poisson": 0.3, "modulus": 100, "wavelength_SPring8": 0.154, "temp": 298}
        logging : function
            ログを出力する関数 デフォルトはprint
        """
        self.poisson = status["poisson"]
        self.modulus = status["modulus"]
        self.wavelength_SPring8 = status["wavelength_SPring8"]
        self.temp = status.get("temp")  # temp を設定 (存在しない場合は None)
        self.is_paras = True
        return logging("パラメータをセットしました")

    def append(self, radial, intensity, psi, logging=print):
        """
        元データを追加する関数

        Parameters
        ----------
        radial : list
            2θのリスト
        intensity : list
            強度のリスト
        psi : float
            追加するデータのψ値
        logging : function
            ログを出力する関数 デフォルトはprint
        """
        intensity = norm(intensity)
        if not self.is_data:
            self.df = pd.DataFrame({"2θ": radial, f"psi={psi}": intensity})
            self.xmin = self.df["2θ"].min()
            self.xmax = self.df["2θ"].max()
            self.is_data = True
        else:
            if self.df is not None:
                self.df[f"psi={psi}"] = intensity
            else:
                # If df is None, create a new DataFrame
                self.df = pd.DataFrame({"2θ": radial, f"psi={psi}": intensity})
                self.xmin = self.df["2θ"].min()
                self.xmax = self.df["2θ"].max()
                self.is_data = True
        self.psi_SPring8.append(psi)
        return logging("データを追加しました")

    def fitting(self, coding=True, logging=print):
        """
        フィッティングを行う関数 アプリに組み込む場合，coding=Falseで確認を無効にする

        Parameters
        ----------
        coding : bool
            coding内での使用を想定しているか
        logging : function
            ログを出力する関数 デフォルトはprint
        """
        # データの有無の確認
        if not self.is_data or self.df is None:
            raise ValueError("データが読み込まれていません")
        if self.is_fitting:  # フィッティング済みの場合は初期化
            if coding:
                print("フィッティング済みです 再フィッティングしますか？")
                ans = input()
                if ans == "n":
                    return logging("キャンセルしました")
            self.paras = np.array([])
            self.twothetas = np.array([])

        # フィッティング
        paras_list = []
        for column in self.df.columns[1:]:
            paras_list.append(gauss_fitting(self.df["2θ"], self.df[column]))
        self.paras = np.array(paras_list)
        self.twothetas = self.paras[:, 1]
        self.is_fitting = True
        return logging("フィッティングが完了しました"), self.calc_stress(
            logging=logging
        )

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
        if self.poisson is None:
            raise ValueError("ポアソン比が設定されていません")
        if self.modulus is None:
            raise ValueError("弾性率が設定されていません")

        # boolsの確認
        if not self.bools:
            self.bools = [True] * len(self.psi_SPring8)
        if len(self.bools) != len(self.psi_SPring8):
            raise ValueError("boolsの長さが不正です")

        # d0の計算
        if self.wavelength_SPring8 is None:
            raise ValueError("wavelength_SPring8 が設定されていません")
        lam = self.wavelength_SPring8 * 1e-10  # 波長 (m)
        v = self.poisson  # ポアソン比
        sin2psi = np.sin(np.radians(self.psi_SPring8)) ** 2
        d = lam / (
            2 * np.sin(np.radians(np.array(self.twothetas) / 2))
        )  # Braggの式からdを計算 (m)
        sin2psi_calc = sin2psi[self.bools]  # sin2ψの計算
        d_calc = d[self.bools]  # dの計算
        a1, b1 = np.polyfit(
            sin2psi_calc, d_calc, 1
        )  # sin2ψ vs. d の直線のフィッティング
        d0 = b1 + v * a1 / (1 + v)  # sin2ψ=v/(1+v)のときにd0になる (m)
        sintheta0 = lam / (2 * d0)  # sinθ0の計算
        theta0 = np.arcsin(sintheta0)  # θ0の計算

        # 応力の計算
        K = (
            -self.modulus / (2 * (1 + self.poisson)) * np.pi / 180 / np.tan(theta0)
        )  # (MPa/deg.)
        twotheta_calc = np.array(self.twothetas)[self.bools]  # 2θの計算
        M, A = np.polyfit(sin2psi_calc, twotheta_calc, 1)  # M (deg.)
        r = K * M  # (GPa)
        r = r * 1e3  # (MPa)

        # 信頼区間の計算
        n = len(sin2psi_calc)  # データ数
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
        y_hat = M * sin2psi_calc + A  # 予測値
        s = twotheta_calc - y_hat  # 残差
        s2 = np.sum(s**2) / (n - 2)  # 残差分散
        se = np.sqrt(s2 / np.sum((sin2psi_calc - np.mean(sin2psi_calc)) ** 2))
        ci = t * se  # 信頼区間係数
        ci_r = ci * K  # 応力の信頼区間
        ci_r = abs(ci * K)  # 応力の信頼区間の絶対値 (GPa)
        ci_r = ci_r * 1e3  # 応力の信頼区間の絶対値 (MPa)

        # e vs. sin2ψ の直線の傾きと切片
        e = d / d0 - 1
        e_calc = e[self.bools]  # εの計算
        a2, b2 = np.polyfit(sin2psi_calc, e_calc, 1)

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
        応力の再計算を行う関数

        Parameters
        ----------
        bools : list
            測定された各ψにおけるデータを応力計算時に使用するかのリスト
        logging : function
            ログを出力する関数 デフォルトはprint
        """
        self.bools = bools
        self.is_calculated = False
        return self.calc_stress(logging=logging)

    def save_datas(self, path, name, logging=print):
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
        """
        self.name = name
        # ファイル名の作成
        save_path = os.path.join(path, name)
        self.path = save_path
        # フォルダの作成
        Path(save_path).mkdir(parents=True, exist_ok=True)
        # データの保存
        with open(save_path + "/.json", "w") as f:
            json.dump(
                {
                    "poisson": self.poisson,
                    "modulus": self.modulus,
                    "wavelength_SPring8": self.wavelength_SPring8,
                    "psi_SPring8": self.psi_SPring8,
                    "temp": self.temp,  # temp を保存
                    "paras": self.paras.tolist(),
                    "twothetas": self.twothetas.tolist(),
                    "a1": self.a1,
                    "b1": self.b1,
                    "a2": self.a2,
                    "b2": self.b2,
                    "d0": self.d0,
                    "d": self.d.tolist(),
                    "e": self.e.tolist(),
                    "r": self.r,
                    "sin2psi": self.sin2psi.tolist(),
                    "name": self.name,
                    "ci_r": self.ci_r,
                    "bools": self.bools,
                    "kind": self.kind,
                    "is_saved": self.is_saved,
                    "is_calculated": self.is_calculated,
                    "is_fitting": self.is_fitting,
                    "is_data": self.is_data,
                    "is_paras": self.is_paras,
                    "path": self.path,
                    "xmin": self.xmin,
                    "xmax": self.xmax,
                },
                f,
                indent=4,
            )
        if self.df is not None:
            self.df.to_csv(save_path + "/.csv", index=False)
        self.is_saved = True
        return logging("データを保存しました")

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
        plt.rcParams["svg.fonttype"] = "none"
        # データの有無の確認
        if not self.is_data or self.df is None:
            raise ValueError("データが読み込まれていません")

        # パラメータの確認
        label = [f"ψ={psi}" for psi in self.psi_SPring8]

        # プロット
        for i, column in enumerate(self.df.columns[1:]):
            plt.plot(
                self.df["2θ"],
                self.df[column] + i * 0.2,
                label=label[i],
                color=cm.GnBu((i + 5) / (len(self.df.columns) + 4)), # type: ignore
            )
        # プロットの詳細設定
        plt.xlim(self.xmin, self.xmax)
        plt.xticks(np.arange(self.xmin, self.xmax + 0.1, 0.2))
        plt.yticks([])
        plt.xlabel("2θ")
        plt.ylabel("Intensity")
        plt.legend()
        if save_path:
            plt.savefig(save_path)
            plt.show()
            return logging("プロットを保存しました")
        plt.show()

    def plot_fitting(self, save_path=None, logging=print, ax=None):
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
        plt.rcParams["svg.fonttype"] = "none"
        # データの有無の確認
        if not self.is_data:
            raise ValueError("データが読み込まれていません")
        if not self.is_fitting:
            raise ValueError("フィッティングが行われていません")

        # パラメータの確認
        label = [f"ψ={psi}" for psi in self.psi_SPring8]

        # プロットの初期化
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))

        # プロット
        for i, column in enumerate(self.df.columns[1:]):  # 生データ
            ax.plot(
                self.df["2θ"],
                self.df[column] + i * 0.2,
                label=label[i],
                color=cm.GnBu((i + 5) / (len(self.df.columns) + 4)),
            )
        for i, para in enumerate(self.paras):  # フィッティング
            ax.plot(
                self.df["2θ"],
                gauss(self.df["2θ"], *para) + i * 0.2,
                color=cm.Oranges((i + 5) / (len(self.df.columns) + 4)),
            )
        for i, twotheta in enumerate(self.twothetas):  # ピーク
            ax.plot(twotheta, 0.9 + i * 0.2, "^k")

        # プロットの詳細設定
        plt.xlim(self.xmin, self.xmax)
        plt.xticks(np.arange(self.xmin, self.xmax + 0.1, 0.2))
        plt.yticks([])
        plt.xlabel("2θ")
        plt.ylabel("Intensity")
        if ax is None:
            plt.legend()
            return plt.show()
        if save_path:
            plt.savefig(save_path)
            plt.show()
            return logging("プロットを保存しました")

        return ax

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
        plt.rcParams["svg.fonttype"] = "none"
        # データの有無の確認
        if not self.is_calculated:
            raise ValueError("計算が行われていません")

        # プロット
        x = np.array([0, 0.8])
        d = self.a1 * x + self.b1
        plt.plot(x, d, "--")
        plt.plot(self.sin2psi, self.d, "o")
        # プロットの詳細設定
        plt.xlim(0, 0.8)
        plt.xlabel("sin$^2\psi$")
        plt.ylabel("d [nm]")
        if save_path:
            plt.savefig(save_path)
            plt.show()
            return logging("プロットを保存しました")
        plt.show()

    def plot_e(self, save_path=None, logging=print, ax=None):
        """
        ε vs. sin2ψ のプロットを行う関数
        coding専用関数

        Parameters
        ----------
        save_path : str
            保存先のパス デフォルトはNone(保存しない)
        logging : function
            ログを出力する関数 デフォルトはprint
        ax : matplotlib.axes.Axes
            プロットを描画する既存のAxesオブジェクト デフォルトはNone(新規作成)
        """
        plt.rcParams["svg.fonttype"] = "none"
        # データの有無の確認
        if not self.is_calculated:
            raise ValueError("計算が行われていません")

        # プロットの初期化
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))

        # プロット
        x = np.array([0, 1])
        e = self.a2 * x + self.b2
        ax.plot(x, e, "--")
        ax.plot(self.sin2psi, self.e, "o", markersize=15, color="white", markeredgecolor="black", markeredgewidth=1.5)

        # プロットの詳細設定
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.003, 0.003)
        ax.set_xlabel("sin$^2\psi$")
        ax.set_ylabel("ε (%)")

        if save_path:
            plt.savefig(save_path)
            plt.show()
            return logging("プロットを保存しました")

        if ax is None:
            plt.show()

        return ax
