import os
import json
import time
import pickle
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit as cf
import chardet


# Progress tracking and result collection system
class ProcessingResult:
    """処理結果を収集・管理するクラス"""

    def __init__(self):
        self.successes = []
        self.errors = []
        self.warnings = []
        self.start_time = time.time()

    def add_success(self, item, stress=None, ci_stress=None, details=None):
        """成功した処理を記録"""
        self.successes.append(
            {
                "file": item,
                "stress": stress,
                "ci_stress": ci_stress,
                "details": details,
                "timestamp": time.time(),
            }
        )

    def add_error(self, item, error):
        """エラーを記録"""
        self.errors.append(
            {"file": item, "error": str(error), "timestamp": time.time()}
        )

    def add_warning(self, item, warning):
        """警告を記録"""
        self.warnings.append(
            {"file": item, "warning": str(warning), "timestamp": time.time()}
        )

    def print_summary(self, verbose=False):
        """結果サマリーを表示"""
        elapsed = time.time() - self.start_time
        print(f"\n📊 処理完了 ({elapsed:.1f}秒)")
        print(f"✅ 成功: {len(self.successes)}件")
        print(f"⚠️ 警告: {len(self.warnings)}件")
        print(f"❌ エラー: {len(self.errors)}件")

        # 成功したサンプルの表示
        if self.successes:
            print(f"\n成功したサンプル:")
            display_count = min(5, len(self.successes))
            for success in self.successes[:display_count]:
                filename = success["file"]
                if success["stress"] is not None and success["ci_stress"] is not None:
                    print(
                        f"  - {filename}: 残留応力 {success['stress']:.1f} ± {success['ci_stress']:.1f} MPa"
                    )
                else:
                    print(f"  - {filename}: 処理完了")

            if len(self.successes) > display_count:
                print(f"  ... 他{len(self.successes) - display_count}件")

        # 警告の表示
        if self.warnings and verbose:
            print(f"\n⚠️ 警告詳細:")
            for warning in self.warnings:
                print(f"  - {warning['file']}: {warning['warning']}")

        # エラーの表示
        if self.errors:
            print(f"\n❌ エラー詳細:")
            for error in self.errors:
                print(f"  - {error['file']}: {error['error']}")

        if self.successes:
            print(f"\n処理されたデータはdatas/SmartLabに保存されました")

    def get_summary_dict(self):
        """サマリー情報を辞書で返す"""
        return {
            "total_processed": len(self.successes) + len(self.errors),
            "successes": len(self.successes),
            "warnings": len(self.warnings),
            "errors": len(self.errors),
            "processing_time": time.time() - self.start_time,
            "success_files": [s["file"] for s in self.successes],
            "error_files": [e["file"] for e in self.errors],
        }


class AnalysisUtils:
    """解析用の数値計算・ユーティリティ関数を提供するクラス"""

    @staticmethod
    def norm(data: np.ndarray) -> np.ndarray:
        """データの正規化 (0-1)"""
        min_v = np.min(data)
        max_v = np.max(data)
        if max_v == min_v:
            return np.zeros_like(data)
        return (data - min_v) / (max_v - min_v)

    @staticmethod
    def gauss(x, height, center, sigma, base):
        """ガウス関数"""
        return base + height * np.exp(-((x - center) ** 2) / (2 * sigma**2))

    @staticmethod
    def gauss_fitting(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        """ガウス関数によるフィッティング"""
        x = []
        y = []
        # 0.5以上のデータのみを取り出す（ピーク付近）
        threshold = 0.5
        mask = yy >= threshold
        x = xx[mask]
        y = yy[mask]

        if len(x) == 0:
            x = xx
            y = yy

        if len(y) > 0:
            x_max = x[np.argmax(y)]
        else:
            x_max = np.mean(xx)

        try:
            # 初期値: height=1, center=x_max, sigma=0.1, base=0
            popt, _ = cf(
                AnalysisUtils.gauss, x, y, p0=[1, x_max, 0.1, 0], maxfev=100000
            )
            return popt
        except Exception:
            # フィッティング失敗時のデフォルト値
            return np.array([0, x_max, 1, 0])

    @staticmethod
    def calculate_stress_parameters(twothetas, psis, modulus, poisson, wavelength):
        """応力計算のコアロジック"""
        E = modulus
        v = poisson
        lam = wavelength
        sin2psi = np.sin(np.radians(psis)) ** 2

        # 応力の計算
        # Braggの式: 2d sin(theta) = n * lambda
        # theta = 2theta / 2
        d = lam / (2 * np.sin(np.radians(np.array(twothetas) / 2)))

        # sin2psi vs d
        a1, b1 = np.polyfit(sin2psi, d, 1)
        d0 = b1 + v * a1 / (1 + v)

        # ひずみ epsilon
        e = (d - d0) / d0

        # sin2psi vs epsilon
        a2, b2 = np.polyfit(sin2psi, e, 1)

        # 応力 sigma (MPa)
        r = a2 * E / (1 + v) * 1000

        return {
            "d": d,
            "e": e,
            "r": r,
            "d0": d0,
            "a1": a1,
            "b1": b1,
            "a2": a2,
            "b2": b2,
            "sin2psi": sin2psi,
        }


class SmartLabDataLoader:
    """SmartLabデータの読み込みを担当するクラス"""

    @staticmethod
    def read_data(file_path: str, encoding: Optional[str] = None) -> pd.DataFrame:
        """
        SmartLabのデータを読み込む
        TXTファイルとASCファイルに対応
        """
        _, ext = os.path.splitext(file_path.lower())

        if ext == ".asc":
            return SmartLabDataLoader._read_asc_data(file_path)
        elif ext == ".txt":
            return SmartLabDataLoader._read_txt_data(file_path, encoding)
        else:
            raise ValueError(f"サポートされていないファイル形式です: {ext}")

    @staticmethod
    def _read_asc_data(file_path: str) -> pd.DataFrame:
        xrd = []
        with open(file_path, "r") as f:
            txt = f.read()
            for block in txt.split("BEGIN")[1:]:
                block = block.split("END")[0]
                lines = block.split("\n")[1:-1]

                # パラメータ抽出
                start_line = [l for l in lines if "START" in l][0]
                start = float(start_line.split("=")[1])
                stop_line = [l for l in lines if "STOP" in l][0]
                stop = float(stop_line.split("=")[1])
                step_line = [l for l in lines if "STEP" in l][0]
                step = float(step_line.split("=")[1])

                twotheta = np.arange(start, stop + 0.0001, step)
                data_lines = [l for l in lines if "*" not in l]
                data = [float(d) for line in data_lines for d in line.split(",")]
                xrd.append((twotheta, data))

        # 正規化
        xrd = [(t, AnalysisUtils.norm(np.array(d))) for t, d in xrd]

        # データ整合性チェック
        if len(set([len(t) for t, _ in xrd])) != 1:
            raise ValueError("2θのデータ数が異なります")

        df = pd.DataFrame(xrd[0][0], columns=["2θ"])
        for i, (_, data) in enumerate(xrd, start=1):
            df[f"real_{i}"] = data

        return df

    @staticmethod
    def _read_txt_data(file_path: str, encoding: Optional[str] = None) -> pd.DataFrame:
        if not encoding:
            with open(file_path, "rb") as f:
                result = chardet.detect(f.read())
                encoding = result["encoding"]

        data = []
        current_data = []
        in_data_section = False

        with open(file_path, "r", encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line[0].isdigit():
                    values = line.split()
                    if len(values) >= 2:
                        try:
                            float(values[0])
                            float(values[1])
                            current_data.append([float(values[0]), float(values[1])])
                            in_data_section = True
                            continue
                        except ValueError:
                            pass

                if in_data_section and not line[0].isdigit():
                    if current_data:
                        data.append(current_data)
                        current_data = []
                    in_data_section = False

            if current_data:
                data.append(current_data)

        if not data:
            raise ValueError("有効な数値データが見つかりませんでした")

        df = pd.DataFrame(data[0], columns=["2θ", "real_1"])

        for i, dataset in enumerate(data[1:], start=2):
            temp_df = pd.DataFrame(dataset, columns=["2θ", f"real_{i}"])

            # 簡易チェック
            if len(df) != len(temp_df):
                raise ValueError(
                    f"データ長が一致しません: real_1={len(df)}, real_{i}={len(temp_df)}"
                )

            df = pd.merge(df, temp_df, on="2θ", how="outer")

        # 正規化
        for col in df.columns[1:]:
            df[col] = AnalysisUtils.norm(df[col].values)

        return df


class SmartLab_data:
    """
    SmartLabのデータを管理するクラス
    """

    # T分布のt値 (自由度(n-2), 信頼率(1-α) = 0.683)
    T_DISTRIBUTION = {
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
    }

    def __init__(self, path: Optional[str] = None):
        self.path = path

        # パラメータ
        self.poisson: Optional[float] = None
        self.modulus: Optional[float] = None
        self.wavelength_SmartLab: Optional[float] = None
        self.psi_SmartLab: List[float] = []
        self.is_paras: bool = False

        # 元データ
        self.df: Optional[pd.DataFrame] = None
        self.xmin: Optional[float] = None
        self.xmax: Optional[float] = None
        self.is_data: bool = False

        # フィッティング結果
        self.paras: List[np.ndarray] = []
        self.twothetas: np.ndarray = np.array([])
        self.is_fitting: bool = False

        # 応力の計算結果
        self.a1: Optional[float] = None
        self.b1: Optional[float] = None
        self.a2: Optional[float] = None
        self.b2: Optional[float] = None
        self.d0: Optional[float] = None
        self.d: List[float] = []
        self.e: List[float] = []
        self.r: Optional[float] = None
        self.sin2psi: List[float] = []
        self.is_calculated: bool = False
        self.bools: List[bool] = []
        self.ci_r: Optional[float] = None

        # 名前
        self.name: Optional[str] = None
        self.is_saved: bool = False
        self.kind: str = "single"

        if path:
            self.load_from_path(path)

    def load_from_path(self, path: str):
        """保存済みデータを読み込む"""
        try:
            with open(os.path.join(path, ".json"), "r") as f:
                data = json.load(f)

            self.poisson = data.get("poisson")
            self.modulus = data.get("modulus")
            self.wavelength_SmartLab = data.get("wavelength_SmartLab")
            self.psi_SmartLab = data.get("psi_SmartLab", [])
            self.is_paras = data.get("is_paras", False)

            csv_path = os.path.join(path, ".csv")
            if os.path.exists(csv_path):
                self.df = pd.read_csv(csv_path)
                self.xmin = self.df["2θ"].min()
                self.xmax = self.df["2θ"].max()

            self.is_data = data.get("is_data", False)
            self.paras = np.array(data.get("paras", []))
            self.twothetas = np.array(data.get("twothetas", []))
            self.is_fitting = data.get("is_fitting", False)

            self.a1 = data.get("a1")
            self.b1 = data.get("b1")
            self.a2 = data.get("a2")
            self.b2 = data.get("b2")
            self.d0 = data.get("d0")
            self.d = data.get("d", [])
            self.e = data.get("e", [])
            self.r = data.get("r")
            self.sin2psi = data.get("sin2psi", [])
            self.is_calculated = data.get("is_calculated", False)
            self.bools = data.get("bools", [])
            self.ci_r = data.get("ci_r")

            self.name = data.get("name")
            self.is_saved = data.get("is_saved", False)

        except Exception as e:
            print(f"データの読み込みに失敗しました: {e}")

    def set_status(self, status, logging=print):
        """
        各物性値を設定する関数

        Parameters
        ----------
        status : dict
            物性値を格納した辞書 例: {"poisson": 0.3, "modulus": 100, "wavelength_SmartLab": 0.154, "psi_SmartLab": [0, 45, 90]}
        logging : function
            ログを出力する関数 デフォルトはprint
        """
        self.poisson = status["poisson"]
        self.modulus = status["modulus"]
        self.wavelength_SmartLab = status["wavelength_SmartLab"]
        self.psi_SmartLab = status["psi_SmartLab"]
        self.is_paras = True
        return logging("パラメータをセットしました")

    def read_txt(self, data_path, logging=print, result_collector=None):
        """
        元データのtxtファイルを読み込む関数

        Parameters
        ----------
        data_path : str
            元データのtxtファイルのパス
        logging : function
            ログを出力する関数 デフォルトはprint
        result_collector : ProcessingResult, optional
            処理結果を収集するオブジェクト
        """
        try:
            self.df = SmartLabDataLoader.read_data(data_path)
            self.xmin = self.df["2θ"].min()
            self.xmax = self.df["2θ"].max()
            self.is_data = True

            if result_collector:
                result_collector.add_success(data_path, details="データ読み込み完了")

            logging("データを読み込みました")
            return self.fitting(logging=logging, result_collector=result_collector)

        except Exception as e:
            if result_collector:
                result_collector.add_error(data_path, e)
            raise

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
            if not self.is_data:
                raise ValueError("データが読み込まれていません")
            if self.is_fitting:  # フィッティング済みの場合は初期化
                if coding:
                    ans = input(
                        "フィッティング済みです 再フィッティングしますか？(y/n)"
                    )
                    if ans == "n":
                        return logging("キャンセルしました")
                self.paras = []
                self.twothetas = []

            # フィッティング
            for i in range(1, len(self.df.columns)):
                self.paras.append(
                    AnalysisUtils.gauss_fitting(self.df["2θ"], self.df[f"real_{i}"])
                )
            self.paras = np.array(self.paras)
            self.twothetas = self.paras[:, 1]
            self.is_fitting = True

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

        # d0の計算
        lam = self.wavelength_SmartLab * 1e-10  # 波長 (m)
        v = self.poisson  # ポアソン比
        sin2psi = np.sin(np.radians(self.psi_SmartLab)) ** 2
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
        n = len(self.psi_SmartLab)
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
        if len(bools) != len(self.psi_SmartLab):
            raise ValueError("boolsの長さがpsi_SmartLabの長さと一致しません")

        # 応力の再計算
        thothetas = []
        psis = []
        for i, bool in enumerate(bools):
            if bool:
                thothetas.append(self.twothetas[i])
                psis.append(self.psi_SmartLab[i])
        result = AnalysisUtils.calculate_stress_parameters(
            thothetas,
            psis,
            self.modulus,
            self.poisson,
            self.wavelength_SmartLab,
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

    def save_datas(self, path, name, logging=print, result_collector=None):
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
                        "wavelength_SmartLab": self.wavelength_SmartLab,
                        "psi_SmartLab": self.psi_SmartLab,
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
            self.df.to_csv(save_path + "/.csv", index=False)
            self.is_saved = True

            if result_collector:
                result_collector.add_success(
                    name, stress=self.r, ci_stress=self.ci_r, details="データ保存完了"
                )

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
        # データの有無の確認
        if not self.is_data:
            raise ValueError("データが読み込まれていません")

        # パラメータの確認
        if self.is_paras:
            label = [f"ψ={psi}" for psi in self.psi_SmartLab]
        else:
            label = [f"real_{i}" for i in range(1, len(self.df.columns))]

        # プロット
        for i in range(1, len(self.df.columns)):
            ax.plot(
                self.df["2θ"],
                self.df[f"real_{i}"] + (i - 1) * 0.2,
                label=label[i - 1],
                color=cm.GnBu((i + 4) / (len(self.df.columns) + 4)),
            )
        # プロットの詳細設定
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_xticks(np.arange(self.xmin, self.xmax + 0.1, 0.2))
        ax.set_yticks([])
        ax.set_xlabel("2θ")
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
        if not self.is_data:
            raise ValueError("データが読み込まれていません")
        if not self.is_fitting:
            raise ValueError("フィッティングが行われていません")

        # パラメータの確認
        if self.is_paras:
            label = [f"ψ={psi}" for psi in self.psi_SmartLab]
        else:
            label = [f"real_{i}" for i in range(1, len(self.df.columns))]

        # プロット
        for i in range(1, len(self.df.columns)):  # 生データ
            ax.plot(
                self.df["2θ"],
                self.df[f"real_{i}"] + (i - 1) * 0.2,
                label=label[i - 1],
                color=cm.GnBu((i + 4) / (len(self.df.columns) + 4)),
            )
        for i, para in enumerate(self.paras):  # フィッティング
            ax.plot(
                self.df["2θ"],
                AnalysisUtils.gauss(self.df["2θ"], *para) + i * 0.2,
                color=cm.Oranges((i + 4) / (len(self.df.columns) + 4)),
            )
        for i, twotheta in enumerate(self.twothetas):  # ピーク
            ax.plot(twotheta, 0.9 + i * 0.2, "^k")
        # プロットの詳細設定
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_xticks(np.arange(self.xmin, self.xmax + 0.1, 0.2))
        ax.set_yticks([])
        ax.set_xlabel("2θ")
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
        ax.set_xlabel("sin$^2\psi$")
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
        ax.set_xlabel("sin$^2\psi$")
        ax.set_ylabel("ε")
        plt.tight_layout()
        if save_path:
            # 拡張子を.svgに強制
            base, ext = os.path.splitext(save_path)
            save_path_svg = base + ".svg"
            plt.savefig(save_path_svg)
            print(f"プロットを {save_path_svg} に保存しました")

        # --- ここからWebアプリ用のメソッドを追加 ---

    def get_xrd_data(self):
        """フロントエンド(Chart.js)に渡すためのXRDプロット用データを生成して返す"""
        if (
            not self.is_fitting
            or self.xmin is None
            or self.xmax is None
            or self.df is None
        ):
            return []

        plot_data = []
        x_range = np.linspace(self.xmin, self.xmax, 200)

        for i in range(len(self.psi_SmartLab)):
            raw_data = self.df[["2θ", f"real_{i+1}"]].copy()
            raw_data.columns = ["x", "y"]

            # 生データにオフセットを追加
            offset = i * (raw_data["y"].max() - raw_data["y"].min()) * 0.2
            raw_data["y"] += offset

            # フィッティング曲線にオフセットを追加
            fit_y = AnalysisUtils.gauss(x_range, *self.paras[i]) + offset

            series_data = {
                "psi": self.psi_SmartLab[i],
                "raw_data": raw_data.to_dict("records"),
                "fit_curve": [{"x": x, "y": y} for x, y in zip(x_range, fit_y)],
                "peak": {
                    "x": self.twothetas[i],
                    "y": AnalysisUtils.gauss(self.twothetas[i], *self.paras[i])
                    + offset
                    + 0.1,
                },
            }
            plot_data.append(series_data)

        return plot_data

    def get_sin2psi_data(self):
        """フロントエンド(Chart.js)に渡すためのε-sin²ψプロット用データを生成して返す"""
        if not self.is_calculated:
            return {}

        measured_points = [
            {"x": s, "y": e_val} for s, e_val in zip(self.sin2psi, self.e)
        ]

        # フィッティング直線の描画範囲を決定
        x_min_fit = 0
        x_max_fit = max(self.sin2psi) * 1.1 if any(self.sin2psi) else 0.8
        x_fit = np.array([x_min_fit, x_max_fit])
        y_fit = self.a2 * x_fit + self.b2

        fit_line = [{"x": x, "y": y} for x, y in zip(x_fit, y_fit)]

        return {"measured": measured_points, "fit": fit_line}


class SmartLab_datas:
    """
    SmartLabの複数データを管理するクラス

    Attributes
    ----------
    name : str
        データの名前
    datas : list
        SmartLab_dataのリスト
    r : list
        応力のリスト
    r_avg : float
        応力の平均値
    r_stdev : float
        応力の標準偏差
    kind : str
        SmartLab_datasで使用する際に，このデータが単一サンプルのものか複数の平均データかを示す このクラスでは"multi"
    is_saved : bool
        データが保存されているか

    Methods
    -------
    add_data(data, logging=print)
        データを追加する関数
    check_datas(logging=print)
        データ一覧を確認する関数
    delete_data(index, coding=True, logging=print)
        各データを削除する関数
    save_datas(path, name, logging=print)
        データを保存する関数
    plot(save_path=None, logging=print)
        クラス内に保存された各データの応力値をプロットする関数

    See Also
    --------
    SmartLab_data : SmartLabのデータを管理するクラス
    SmartLab360_data : SmartLabでの360°3軸応力測定のデータを管理するクラス
    """

    def __init__(self):
        """
        コンストラクタ
        """
        self.name = None
        self.datas = []
        self.r = []
        self.r_avg = None
        self.r_stdev = None
        self.kind = "multi"
        self.is_saved = False

    # データの追加
    def add_data(self, data, logging=print):
        """
        データを追加する関数

        Parameters
        ----------
        data : SmartLab_data or SmartLab_datas
            追加するデータ
        logging : function
            ログを出力する関数 デフォルトはprint
        """

        # 保存されているか確認  名前を付けている場合のみ追加したい
        if not data.is_saved:
            raise ValueError(
                "データが保存されていません データを一度保存してから追加してください"
            )
        # データの追加
        self.datas.append(data)
        # 応力の追加
        if data.kind == "single":
            self.r.append(data.r)
        elif data.kind == "multi":
            self.r.extend(data.r)
        # 応力の計算
        self.r_avg = np.mean(self.r)
        self.r_stdev = np.std(self.r)
        return logging("データを追加しました")

    # データの確認
    def check_datas(self, logging=print):
        """
        データ一覧を確認する関数

        Parameters
        ----------
        logging : function
            ログを出力する関数 デフォルトはprint
        """
        for i, data in enumerate(self.datas):
            logging(i, ":", data.name, data.kind)

    # データの削除
    def delete_data(self, index, coding=True, logging=print):
        """
        データの削除を行う関数 アプリに組み込む場合，coding=Falseで確認を無効にする

        Parameters
        ----------
        index : int
            削除するデータのインデックス check_datasで確認できる
        coding : bool
            coding内での使用を想定しているか
        logging : function
            ログを出力する関数 デフォルトはprint
        """
        # 削除するか確認
        if coding:
            logging("以下のデータを削除します 本当によろしいですか？ (y/n)")
            logging(self.datas[index].name)
            ans = input()
            if ans == "n":
                return logging("キャンセルしました")

        # データの削除
        self.datas.pop(index)
        self.r.pop(index)
        self.r_avg = np.mean(self.r)
        self.r_stdev = np.std(self.r)
        return logging("データを削除しました")

    # データの保存
    def save_datas(self, path, name, logging=print):
        """
        データを保存する関数

        Parameters
        ----------
        path : str
            保存先のフォルダのパス
        name : str
            ファイル名
        logging : function
            ログを出力する関数 デフォルトはprint
        """
        # ファイル名の作成
        save_path = os.path.join(path, name)
        # データの保存  pickleで保存
        with open(save_path + ".pkl", "wb") as f:
            pickle.dump(self, f)
        self.is_saved = True
        return logging("データを保存しました")

    # coding専用関数
    def plot(self, save_path=None, logging=print):
        """
        データをプロットする関数
        coding専用関数
        """
        plt.rcParams["svg.fonttype"] = "none"
        # プロット
        for i, data in enumerate(self.datas):
            if data.kind == "single":
                plt.bar(data.name, data.r)
            elif data.kind == "multi":
                data.plot()
        # プロットの詳細設定
        plt.xlabel("Data")
        plt.ylabel("Stress [MPa]")
        if save_path:
            plt.savefig(save_path)
            plt.show()
            return logging("プロットを保存しました")
        plt.show()


class SmartLab360_data:
    """
    SmartLabでの360°3軸応力測定のデータを管理するクラス
    """

    def __init__(self, path=None):
        pass

    def read_txt(self, data_path, logging=print):
        """
        データを読み込む関数

        Parameters
        ----------
        data_path : str
            データのパス
        """
        self.df = SmartLabDataLoader.read_data(data_path)
        self.xmin = self.df["2θ"].min()
        self.xmax = self.df["2θ"].max()
        self.is_data = True
        return logging("データを読み込みました"), self.fitting()

    def fitting(self, coding=True, logging=print):
        """
        フィッティングを行う関数
        coding内でないならcheck=Falseで確認を行わない
        """
        # データの有無の確認
        if not self.is_data:
            raise ValueError("データが読み込まれていません")
        if self.is_fitting:  # フィッティング済みの場合は初期化
            if coding:
                logging("フィッティング済みです 再フィッティングしますか？")
                ans = input()
                if ans == "n":
                    return logging("キャンセルしました")
            self.paras = []
            self.twothetas = []

        # フィッティング    フィッティング自体はパラメータがなくてもできる
        for i in range(1, len(self.df.columns)):
            self.paras.append(
                AnalysisUtils.gauss_fitting(self.df["2θ"], self.df[f"real_{i}"])
            )
        self.paras = np.array(self.paras)
        self.twothetas = self.paras[:, 1]
        self.is_fitting = True
        return logging("フィッティングが完了しました"), self.calc_stress()

    def calc_stress(self, logging=print):
        """
        応力を計算する関数
        """
        # パラメータの確認
        if not self.is_paras:
            raise ValueError("パラメータが設定されていません")

        # データの長さ
        len_psi = len(self.psi_SmartLab)
        len_phi = len(self.phi_SmartLab)
        len_twothetas = len(self.twothetas)
        if len_psi * len_phi != len_twothetas:
            raise ValueError(
                "データの数が一致しません 2θの数とpsi, phiの数が一致しているか確認してください"
            )

        # 結晶面間隔の計算
        d = self.wavelength_SmartLab / (
            2 * np.sin(np.radians(np.array(self.twothetas) / 2))
        )
        psi_phi = []
        for psi in self.psi_SmartLab:
            for phi in self.phi_SmartLab:
                psi_phi.append((psi, phi))

        # 応力の計算
        p0 = (0, 0, 0, 0, 0, 0, 0)
        result = cf(
            self.basic_formula,
            psi_phi,
            d,
            p0,
        )
        (
            self.sigma11,
            self.sigma22,
            self.sigma33,
            self.sigma12,
            self.sigma13,
            self.sigma23,
            self.d0,
        ) = result
        return logging("応力の計算が完了しました")

    # 3軸応力の基礎式
    def basic_formula(
        self, psi_phi, sigma11, sigma22, sigma33, sigma12, sigma13, sigma23, d0
    ):
        """
        3軸応力の基礎式

        Parameters
        ----------
        psi_phi : tuple
            (psi, phi)
        sigma11 : float
            σ11
        sigma22 : float
            σ22
        sigma33 : float
            σ33
        sigma12 : float
            σ12
        sigma13 : float
            σ13
        sigma23 : float
            σ23
        d0 : float
            d0

        Returns
        -------
        float
            結晶面間隔
        """
        # 係数
        S2_per_2 = (1 + self.poisson) / self.modulus
        S1 = -self.poisson / self.modulus

        # 変数
        psi, phi = psi_phi
        psi = np.radians(psi)
        phi = np.radians(phi)

        # ひずみの計算
        A1 = (
            sigma11 * np.cos(phi) ** 2
            + sigma12 * np.sin(2 * phi)
            + sigma22 * np.sin(phi) ** 2
            - sigma33
        )
        A2 = sigma11 + sigma22 + sigma33
        A3 = sigma13 * np.cos(phi) + sigma23 * np.sin(phi)
        e = (
            S2_per_2 * A1 * np.sin(psi) ** 2
            + S2_per_2 * sigma33
            + S1 * A2
            + S2_per_2 * A3 * np.sin(2 * psi)
        )
        d = d0 * (1 + e)
        return d


def make_status(
    path,
    name,
    poisson=None,
    modulus=None,
    wavelength_SmartLab=None,
    psi_SmartLab=None,
    phi_SmartLab=None,
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
    wavelength_SmartLab : float
        SmartLabの波長
    psi_SmartLab : list
        SmartLabのpsiのリスト
        psi...手前側に傾ける角度
    phi_SmartLab : list
        SmartLabのphiのリスト
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
                "wavelength_SmartLab": wavelength_SmartLab,
                "psi_SmartLab": psi_SmartLab,
                "name": name,
                "phi_SmartLab": phi_SmartLab,
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
    wavelength_SmartLab=None,
    psi_SmartLab=None,
    phi_SmartLab=None,
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
    wavelength_SmartLab : float
        SmartLabの波長
    psi_SmartLab : list
        SmartLabのpsiのリスト
    phi_SmartLab : list
        SmartLabのphiのリスト
    """
    # データの読み込み
    with open(path + ".json", "r") as f:
        data = json.load(f)
    # データの編集
    if poisson:
        data["poisson"] = poisson
    if modulus:
        data["modulus"] = modulus
    if wavelength_SmartLab:
        data["wavelength_SmartLab"] = wavelength_SmartLab
    if psi_SmartLab:
        data["psi_SmartLab"] = psi_SmartLab
    if phi_SmartLab:
        data["phi_SmartLab"] = phi_SmartLab
    # データの保存
    with open(path + ".json", "w") as f:
        json.dump(data, f, indent=4)
    return (
        logging("statusファイルを編集しました"),
        logging("ポアソン比:", data["poisson"]),
        logging("結晶弾性率:", data["modulus"]),
        logging("SmartLabの波長:", data["wavelength_SmartLab"]),
        logging("SmartLabのψ:", data["psi_SmartLab"]),
        logging("SmartLabのφ:", data["phi_SmartLab"]),
    )


# Batch processing function
def batch_process_smartlab_files(
    file_paths, status, save_dir="datas/SmartLab", verbose=False, show_progress=True
):
    """
    複数のSmartLabファイルを一括処理する関数

    Parameters
    ----------
    file_paths : list
        処理するファイルパスのリスト
    status : dict
        物性値を格納した辞書（set_statusで使用）
    save_dir : str, optional
        保存先ディレクトリ（デフォルト: "datas/SmartLab"）
    verbose : bool, optional
        詳細ログを表示するか（デフォルト: False）
    show_progress : bool, optional
        プログレスバーを表示するか（デフォルト: True）

    Returns
    -------
    ProcessingResult
        処理結果を収集したオブジェクト
    """
    # tqdmのインポートを試行
    try:
        from tqdm import tqdm

        has_tqdm = True
    except ImportError:
        has_tqdm = False
        if show_progress:
            print("tqdmがインストールされていません。プログレスバーは表示されません。")
            print("インストール: pip install tqdm")

    result = ProcessingResult()

    # プログレスバーの初期化
    if has_tqdm and show_progress:
        pbar = tqdm(file_paths, desc="Processing SmartLab files")
        iterator = pbar
    else:
        iterator = file_paths
        if show_progress:
            print(f"Processing {len(file_paths)} files...")

    for i, file_path in enumerate(iterator):
        filename = os.path.basename(file_path)

        try:
            # SmartLab_dataオブジェクトの作成
            data = SmartLab_data()

            # パラメータ設定
            data.set_status(
                status, logging=lambda x: None if not verbose else print(f"  {x}")
            )

            # データ読み込み
            data.read_txt(
                file_path,
                logging=lambda x: None if not verbose else print(f"  {x}"),
                result_collector=result,
            )

            # データ保存
            data.save_datas(
                path=save_dir,
                name=filename,
                logging=lambda x: None if not verbose else print(f"  {x}"),
                result_collector=result,
            )

            # プログレスバーの更新（tqdmがある場合）
            if has_tqdm and show_progress:
                pbar.set_postfix(
                    {"Success": len(result.successes), "Errors": len(result.errors)}
                )
            elif show_progress and not has_tqdm:
                if (i + 1) % 10 == 0 or i == len(file_paths) - 1:
                    print(
                        f"Progress: {i + 1}/{len(file_paths)} (Success: {len(result.successes)}, Errors: {len(result.errors)})"
                    )

        except Exception as e:
            # エラーは既にresult_collectorで収集済み
            if verbose:
                print(f"  ❌ Error processing {filename}: {e}")

    # プログレスバーのクリーンアップ
    if has_tqdm and show_progress:
        pbar.close()

    # 結果サマリーの表示
    result.print_summary(verbose=verbose)

    return result


# Simple batch processing function (without progress bar)
def simple_batch_process(file_paths, status, save_dir="datas/SmartLab"):
    """
    シンプルな一括処理関数（プログレスバーなし）

    Parameters
    ----------
    file_paths : list
        処理するファイルパスのリスト
    status : dict
        物性値を格納した辞書
    save_dir : str, optional
        保存先ディレクトリ

    Returns
    -------
    ProcessingResult
        処理結果
    """
    return batch_process_smartlab_files(
        file_paths=file_paths,
        status=status,
        save_dir=save_dir,
        verbose=False,
        show_progress=False,
    )
