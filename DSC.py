"""
このモジュールは、DSC（示差走査熱量測定）データをExcelまたはASCファイルから解析し、
プロットやガラス転移温度（Tg）検索などを行うためのクラスを提供します。

主な機能:
    - DSCデータの読み込みとセグメント分割
    - 加熱・冷却サイクルごとのプロット作成
    - ガラス転移温度（Tg）の自動検索
    - Chart.js用データ生成

主なクラス:
    DSCAnalyzer: DSCデータの解析・可視化・保存を行うクラス

使用方法:
    DSCAnalyzerクラスをインスタンス化し、plot_dscやget_plot_dataなどのメソッドを利用します。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure
import os
import openpyxl  # Excelファイルの読み込みに必要
from typing import Tuple, List, Optional, Dict, Any, Union
from scipy.signal import savgol_filter

# 定数定義
DEFAULT_MIN_SEGMENT_POINTS = 100
DEFAULT_TEMP_RANGE = (30.0, 200.0)
DEFAULT_WINDOW_LENGTH = 101
DEFAULT_POLYORDER = 3
DATA_MARKER = "#GD"
SEGMENT_TYPES = ["heating", "cooling", "all"]
ORDINAL_NUMBERS = {1: "1st", 2: "2nd", 3: "3rd"}


class DSCAnalyzer:
    """
    DSC（示差走査熱量測定）データをExcelファイルまたはASCファイルから直接解析し、
    プロットを作成するためのクラス。
    """

    def __init__(
        self,
        filepath: str,
        sheet_name: str = "Sheet2",
        min_segment_points: int = DEFAULT_MIN_SEGMENT_POINTS,
    ):
        """
        DSCAnalyzerのインスタンスを初期化します。

        Args:
            filepath: 解析対象のExcelファイルまたはASCファイルのパス
            sheet_name: Excelファイルの場合に読み込むシート名（デフォルト: "Sheet2"）
            min_segment_points: 解析対象とするセグメントの最小データ点数

        Raises:
            FileNotFoundError: 指定されたファイルが存在しない場合
            ValueError: min_segment_pointsが無効な値の場合
        """
        self._validate_inputs(filepath, min_segment_points)

        self.filepath = filepath
        self.sheet_name = sheet_name
        self.min_segment_points = min_segment_points
        self.time: Optional[np.ndarray] = None
        self.temp: Optional[np.ndarray] = None
        self.dsc: Optional[np.ndarray] = None
        self.segments: List[Dict[str, np.ndarray]] = []
        self.tg: Optional[float] = None
        self.weight: Optional[float] = None

        self._load_data()
        self._segment_data()
        self._tg_search()

    def _validate_inputs(self, filepath: str, min_segment_points: int) -> None:
        """入力パラメータを検証します。"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"指定されたファイルが見つかりません: {filepath}")

        if min_segment_points <= 0:
            raise ValueError("min_segment_pointsは正の整数である必要があります")

    def _load_data(self) -> None:
        """
        ファイル拡張子に基づいてExcelファイルまたはASCファイルからデータを読み込み、
        numpy arrayとして整形します。
        """
        _, ext = os.path.splitext(self.filepath)
        ext = ext.lower()

        if ext in [".xlsx", ".xls"]:
            self._load_excel_data()
        elif ext in [".asc"]:
            self._load_asc_data()
        else:
            raise ValueError(f"サポートされていないファイル形式です: {ext}")

    def _load_excel_data(self) -> None:
        """Excelファイルからデータを読み込み、numpy arrayに変換します。"""
        xls = pd.ExcelFile(self.filepath)
        # 指定されたシートが存在するか確認し、なければ最初のシートを試す
        sheet_to_load = (
            self.sheet_name
            if self.sheet_name in xls.sheet_names
            else xls.sheet_names[0]
        )

        found_df = None
        header_index = -1

        try:
            df = pd.read_excel(xls, sheet_name=sheet_to_load, header=None)
            # 'Time(min)'を含むヘッダー行を探す
            for row_idx in range(min(100, len(df))):  # 最初の100行から検索
                row = df.iloc[row_idx]
                for col_idx, cell_value in enumerate(row):
                    if (
                        isinstance(cell_value, str)
                        and cell_value.strip().lower() == "time(min)"
                    ):
                        header_index = row_idx
                        found_df = df
                        break
                if found_df is not None:
                    break
        except Exception as e:
            raise IOError(
                f"ファイル '{os.path.basename(self.filepath)}' のシート '{sheet_to_load}' の読み込みに失敗しました: {e}"
            )

        if found_df is None:
            raise ValueError(
                f"ファイル '{os.path.basename(self.filepath)}' 内のどのシートにも 'Time(min)' ヘッダーが見つかりませんでした。"
            )

        try:
            # ヘッダー行とデータ部分を分離
            header_row = found_df.iloc[header_index]
            data_df = found_df.iloc[header_index + 1 :].copy()
            data_df.columns = header_row

            # 列名を文字列化して整理
            data_df.columns = [
                str(c).strip() if pd.notna(c) else f"Unnamed: {i}"
                for i, c in enumerate(data_df.columns)
            ]

            required_cols = ["Time(min)", "Temp.(℃)", "DSC(mW)"]
            missing_cols = [col for col in required_cols if col not in data_df.columns]
            if missing_cols:
                raise ValueError(
                    f"必要なカラムが見つかりません: {', '.join(missing_cols)}"
                )

            clean_data = (
                data_df[required_cols].apply(pd.to_numeric, errors="coerce").dropna()
            )

            # numpy arrayに変換
            self.time = clean_data["Time(min)"].to_numpy()
            self.temp = clean_data["Temp.(℃)"].to_numpy()
            self.dsc = clean_data["DSC(mW)"].to_numpy()

        except Exception as e:
            raise ValueError(
                f"シート '{sheet_to_load}' のデータ整形中にエラーが発生しました: {e}"
            )

    def _load_asc_data(self) -> None:
        """ASCファイルからデータを読み込み、numpy arrayとして整形します。"""
        try:
            with open(self.filepath, "r", encoding="shift_jis", errors="ignore") as f:
                lines = f.readlines()
        except Exception as e:
            raise IOError(
                f"ファイル '{os.path.basename(self.filepath)}' の読み込みに失敗しました: {e}"
            )

        # データ抽出
        time_list, temp_list, dsc_list, weight = self._extract_data_from_lines(lines)

        if not time_list:
            raise ValueError(
                f"ファイル '{os.path.basename(self.filepath)}' から有効なデータを読み込めませんでした。"
            )

        # numpy arrayに変換
        self.time = np.array(time_list)
        self.temp = np.array(temp_list)
        self.dsc = np.array(dsc_list)

        self.weight = weight

        if self.time.size == 0:
            raise ValueError(
                f"ファイル '{os.path.basename(self.filepath)}' にデータが含まれていません。"
            )

    def _extract_data_from_lines(
        self, lines: List[str]
    ) -> Tuple[List[float], List[float], List[float], float]:
        """ファイルの行からデータを抽出します。"""
        time_list, temp_list, dsc_list = [], [], []

        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith(DATA_MARKER):
                data_part = line[len(DATA_MARKER) + 1 :]  # マーカーとタブを除去
                values = data_part.split("\t")

                if len(values) >= 3:
                    try:
                        time_val = float(values[0])
                        temp_val = float(values[1])
                        dsc_val = float(values[2])
                        time_list.append(time_val)
                        temp_list.append(temp_val)
                        dsc_list.append(dsc_val)
                    except ValueError:
                        continue  # 数値に変換できない行はスキップ

            if i == 20:
                line_s = line.split("\t")
                j = [i for i, val in enumerate(line_s) if "重量" in val][0]
            if i == 21:
                line_s = line.split("\t")
                weight = float(line_s[j])

        return time_list, temp_list, dsc_list, weight

    def _segment_data(self) -> None:
        """温度の昇降サイクルに基づいてデータをセグメントに分割します。"""
        if not self._has_sufficient_data():
            return

        # この時点で配列が存在することが保証されているため、アサーションを追加
        assert self.temp is not None and self.time is not None and self.dsc is not None

        is_heating = self.temp[1] > self.temp[0]
        current_segment_indices = []

        for i in range(len(self.temp) - 1):
            current_segment_indices.append(i)

            if self._is_direction_change(i, is_heating):
                self._add_segment_if_valid(current_segment_indices)
                current_segment_indices = []
                is_heating = not is_heating

        # 最後のセグメントを追加
        current_segment_indices.append(len(self.temp) - 1)
        self._add_segment_if_valid(current_segment_indices)

    def _has_sufficient_data(self) -> bool:
        """十分なデータがあるかチェックします。"""
        return (
            self.temp is not None
            and self.time is not None
            and self.dsc is not None
            and self.temp.size > 0
            and len(self.temp) >= 2
        )

    def _is_direction_change(self, index: int, is_heating: bool) -> bool:
        """温度変化の方向が変わったかチェックします。"""
        assert self.temp is not None  # _segment_dataから呼ばれるため保証されている

        temp_now = self.temp[index]
        temp_next = self.temp[index + 1]

        return (is_heating and temp_next < temp_now) or (
            not is_heating and temp_next > temp_now
        )

    def _add_segment_if_valid(self, indices: List[int]) -> None:
        """有効なセグメントのみを追加します。"""
        assert self.time is not None and self.temp is not None and self.dsc is not None

        if len(indices) >= self.min_segment_points:
            segment_data = {
                "time": self.time[indices],
                "temp": self.temp[indices],
                "dsc": self.dsc[indices],
            }
            self.segments.append(segment_data)

    def _tg_search(
        self,
        temp_range: Tuple[float, float] = DEFAULT_TEMP_RANGE,
        window_length: int = DEFAULT_WINDOW_LENGTH,
        polyorder: int = DEFAULT_POLYORDER,
    ) -> None:
        """
        プロファイルからガラス転移温度を検索します。

        Args:
            temp_range: 解析対象の温度範囲 (min_temp, max_temp)
            window_length: Savitzky-Golayフィルターのウィンドウサイズ
            polyorder: Savitzky-Golayフィルターの多項式次数

        Raises:
            ValueError: セグメントが存在しない、またはデータが不十分な場合
        """
        if not self.segments:
            self.tg = None
            return

        try:
            # 最後のセグメント（通常は2nd heating）を使用
            last_segment = self.segments[-1]
            temp_data = last_segment["temp"]
            dsc_data = last_segment["dsc"]

            # 温度範囲でフィルタリング
            filtered_temp, filtered_dsc = self._filter_by_temperature_range(
                temp_data, dsc_data, temp_range
            )

            if len(filtered_temp) < window_length:
                raise ValueError(
                    f"温度範囲 {temp_range[0]} から {temp_range[1]} のデータが少なすぎます。"
                    f"必要: {window_length}点, 実際: {len(filtered_temp)}点"
                )

            # Savitzky-Golayフィルターで二次微分を計算
            delta_temp = (
                filtered_temp[1] - filtered_temp[0] if len(filtered_temp) > 1 else 1.0
            )
            second_derivative = savgol_filter(
                filtered_dsc,
                window_length=window_length,
                polyorder=polyorder,
                deriv=2,
                delta=delta_temp,
            )

            # 二次微分の最小値位置をTgとする
            peak_index = np.argmin(second_derivative)
            self.tg = float(filtered_temp[peak_index])

        except Exception as e:
            print(f"Tg検索中にエラーが発生しました: {e}")
            self.tg = None

    def _filter_by_temperature_range(
        self,
        temp_data: np.ndarray,
        dsc_data: np.ndarray,
        temp_range: Tuple[float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """指定された温度範囲でデータをフィルタリングします。"""
        mask = (temp_data >= temp_range[0]) & (temp_data <= temp_range[1])
        return temp_data[mask], dsc_data[mask]

    @staticmethod
    def _get_segment_type(segment_dict):
        """セグメントが加熱か冷却かを判断する静的メソッド。"""
        if (
            not segment_dict
            or "temp" not in segment_dict
            or segment_dict["temp"].size == 0
        ):
            return "unknown"
        # 温度データを使用
        temp_data = segment_dict["temp"]
        if temp_data[-1] > temp_data[0]:
            return "heating"
        else:
            return "cooling"

    def plot_dsc(
        self,
        save_path: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = (30, 200),
        ylim: Optional[Tuple[float, float]] = None,
        plot_type: str = "heating",
    ) -> matplotlib.figure.Figure:
        """
        DSCカーブをプロットします。

        Args:
            save_path: プロットを保存するファイルパス
            xlim: X軸の表示範囲
            ylim: Y軸の表示範囲（Noneの場合は自動設定）
            plot_type: プロットする種類 ('heating', 'cooling', 'all')

        Returns:
            matplotlib.pyplot.Figure: 作成されたfigureオブジェクト

        Raises:
            ValueError: plot_typeが無効な値の場合
        """
        if plot_type not in SEGMENT_TYPES:
            raise ValueError(
                f"plot_typeは {SEGMENT_TYPES} のいずれかを指定してください。"
            )

        self._setup_plot_style()
        fig, ax = plt.subplots(figsize=(5, 4))

        heating_count = 0
        cooling_count = 0

        for seg_dict in self.segments:
            seg_type = self._get_segment_type(seg_dict)
            label = self._generate_segment_label(seg_type, heating_count, cooling_count)

            if seg_type == "heating":
                heating_count += 1
            elif seg_type == "cooling":
                cooling_count += 1

            if plot_type == "all" or plot_type == seg_type:
                temp_data = seg_dict["temp"]
                dsc_data = seg_dict["dsc"]
                ax.plot(temp_data, dsc_data, label=label)

        self._configure_axes(ax, xlim, ylim)
        plt.tight_layout()

        if save_path:
            self._save_plot(save_path)

        return fig

    def _setup_plot_style(self) -> None:
        """プロットのスタイルを設定します。"""
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["svg.fonttype"] = "none"
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"

    def _generate_segment_label(
        self, seg_type: str, heating_count: int, cooling_count: int
    ) -> str:
        """セグメントのラベルを生成します。"""
        if seg_type == "heating":
            count = heating_count + 1
            return f"{ORDINAL_NUMBERS.get(count, str(count)+'th')} heat"
        elif seg_type == "cooling":
            count = cooling_count + 1
            return f"{ORDINAL_NUMBERS.get(count, str(count)+'th')} cool"
        return ""

    def _configure_axes(
        self,
        ax: plt.Axes,
        xlim: Optional[Tuple[float, float]],
        ylim: Optional[Tuple[float, float]],
    ) -> None:
        """軸の設定を行います。"""
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Heat Flow (mW)")

        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)

        # 凡例にプロットが1つ以上ある場合のみ表示
        if ax.get_legend_handles_labels()[0]:
            ax.legend()

    def _save_plot(self, save_path: str) -> None:
        """プロットを保存します。"""
        base, ext = os.path.splitext(save_path)
        save_path_svg = base + ".svg"
        plt.savefig(save_path_svg)
        print(f"プロットを {save_path_svg} に保存しました。")

    def get_plot_data(self) -> List[Dict[str, Any]]:
        """
        フロントエンド(Chart.js)に渡すためのプロット用データを生成して返します。

        Returns:
            プロット用のデータ構造のリスト
        """
        if not self.segments:
            return []

        plot_data = []
        heating_count = 0
        cooling_count = 0

        for seg_dict in self.segments:
            seg_type = self._get_segment_type(seg_dict)
            label = self._generate_segment_label(seg_type, heating_count, cooling_count)

            if seg_type == "heating":
                heating_count += 1
            elif seg_type == "cooling":
                cooling_count += 1

            # 辞書から温度データとDSCデータを取得
            temp_data = seg_dict["temp"]
            dsc_data = seg_dict["dsc"]
            data_points = [
                {"x": float(temp), "y": float(dsc)}
                for temp, dsc in zip(temp_data, dsc_data)
            ]

            series_data = {"label": label, "type": seg_type, "data": data_points}
            plot_data.append(series_data)

        return plot_data


if __name__ == "__main__":
    # --- クラスの使用例 ---
    try:
        # 1. 解析したいファイルのパスを指定（ExcelまたはASCファイル）
        # Excelファイルの例
        # file_path = "C:/Users/uttya/OneDrive/神戸大学/西野研/reserch/解析データ/TMA/1_1_PMMA_lam_20250613.xlsx"

        # ASCファイルの例
        file_path = "C:/Users/uttya/OneDrive/神戸大学/西野研/04_Python/app/1_1_EP_none_20250805.ASC"

        # 2. クラスをインスタンス化
        # Excelファイルの場合、sheet_name引数でシート名を指定可能
        # デフォルトで100点未満のセグメントは無視します
        analyzer_dsc = DSCAnalyzer(file_path)

        # 3. プロットを作成
        base_name = os.path.basename(file_path)
        file_name_without_ext = os.path.splitext(base_name)[0]
        save_dir = "C:/Users/uttya/OneDrive/神戸大学/西野研/01_報告会/fig"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # --- ★新しい使い方★ ---
        # 3a. 加熱サイクルのみプロット（デフォルト）
        print("--- 加熱サイクルのみプロット ---")
        output_path_heat = os.path.join(
            save_dir, f"{file_name_without_ext}_DSC_heat.png"
        )
        fig_heat = analyzer_dsc.plot_dsc(
            save_path=output_path_heat, plot_type="heating"
        )
        plt.show()
        plt.close(fig_heat)

        # 3b. 冷却サイクルのみプロット
        print("\n--- 冷却サイクルのみプロット ---")
        output_path_cool = os.path.join(
            save_dir, f"{file_name_without_ext}_DSC_cool.png"
        )
        fig_cool = analyzer_dsc.plot_dsc(
            save_path=output_path_cool, plot_type="cooling"
        )
        plt.show()
        plt.close(fig_cool)

        # 3c. 全てのサイクルをプロット
        print("\n--- 全サイクルをプロット ---")
        output_path_all = os.path.join(save_dir, f"{file_name_without_ext}_DSC_all.png")
        fig_all = analyzer_dsc.plot_dsc(save_path=output_path_all, plot_type="all")
        plt.show()
        plt.close(fig_all)

        print("\nDSC解析が完了しました。")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"エラーが発生しました: {e}")
