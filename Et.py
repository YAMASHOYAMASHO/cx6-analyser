import json
import os
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit as cf


class Et:
    def __init__(self, path: str = None):
        if path:
            self.path = path
            with open(path, "r") as f:
                data = json.load(f)
            self.x = np.array(data["x"])
            self.y = np.array(data["y"])
            self.popt = np.array(data["popt"])
            self.peak = data["peak"]
            self.name = data["name"]
            self.no = data["no"]
            self.stress = data.get("stress")
            self.temp = data.get("temp")
        else:
            self.path = None
            self.x = None
            self.y = None
            self.popt = None
            self.peak = None
            self.name = None
            self.no = None
            self.stress = None
            self.temp = None

    def load(self, xrd_x, xrd_y):
        self.x = xrd_x
        self.y = self.norm(xrd_y)
        self.popt = self.gauss_fitting(self.x, self.y)
        self.peak = self.popt[1]

    def plot_xrd(self, path=None):
        if not self.x or not self.y:
            raise ValueError("XRD data not loaded. Please load the data first.")
        plt.rcParams["svg.fonttype"] = "none"
        plt.rcParams["font.size"] = 18
        plt.plot(self.x, self.y)
        plt.plot(self.x, self.gauss(self.x, *self.popt))
        plt.plot(self.peak, 0.9, "r^")
        plt.xlabel("2θ(deg.)")
        plt.ylabel("Intensity")
        if path:
            plt.savefig(path)

    def norm(self, data):
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

    def gauss(self, x, height, center, sigma, base):
        return base + height * np.exp(-((x - center) ** 2) / (2 * sigma**2))

    def gauss_fitting(self, xx, yy):
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
        x = [x for x, y in zip(xx, yy) if y >= 0.5]
        y = [y for y in yy if y >= 0.5]
        x = np.array(x)
        y = np.array(y)
        # おおよその初期値を設定
        x_max = x[np.argmax(y)]
        # フィッティング
        popt, _ = cf(self.gauss, x, y, p0=[1, x_max, 0.1, 0], maxfev=10000)
        # フィッティングの結果を取得
        return popt

    def save(self, path):
        """
        データを保存する関数

        Parameters
        ----------
        path : str
            保存先のパス
        """
        with open(path, "w") as f:
            data = {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in self.__dict__.items()
            }
            json.dump(data, f, indent=4)
