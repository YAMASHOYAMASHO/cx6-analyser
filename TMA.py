"""
このモジュールは、TMA（熱機械分析）データをExcelまたはASCファイルから解析し、
熱膨張率の計算やプロットを行うためのクラスを提供します。

主な機能:
    - TMAデータの読み込みとセグメント分割
    - 熱膨張率の計算とプロット作成
    - Chart.js用データ生成

主なクラス:
    TMAAnalyzer: TMAデータの解析・可視化・保存を行うクラス

使用方法:
    TMAAnalyzerクラスをインスタンス化し、plot_expansionやget_plot_dataなどのメソッドを利用します。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import openpyxl  # Excelファイルの読み込みに必要
from typing import Tuple, List, Optional, Dict, Any


class TMAAnalyzer:
    """
    TMA（熱機械分析）データを解析し、プロットを作成するためのクラス。
    ExcelファイルとASCファイルの両方に対応。

    Attributes:
        filepath (str): 解析するデータファイルのパス。
        time (np.ndarray): 時間データ。
        temp (np.ndarray): 温度データ。
        tma (np.ndarray): TMAデータ。
        segments (list): 温度サイクルで分割されたデータセグメント（辞書のリスト）。
        expansion_data (list): 各セグメントの温度と計算された熱膨張率のデータ（辞書のリスト）。
    """

    def __init__(
        self,
        filepath: str,
        sheet_name: str = "Sheet2",
        sample_length_mm: float = 20.0,
        min_segment_points: int = 100,
        temp_range_c: Tuple[float, float] = (30.0, 200.0),
    ):
        """
        TMAAnalyzerのインスタンスを初期化します。

        Args:
            filepath (str): 解析対象のExcelファイルまたはASCファイルのパス。
            sheet_name (str, optional): Excelファイルの場合に読み込むシート名。デフォルトは "Sheet2"。
            sample_length_mm (float, optional): サンプルの初期長 (mm)。デフォルトは 20.0。
            min_segment_points (int, optional): 有効と見なすセグメントの最小データ点数。デフォルトは 100。
            temp_range_c (Tuple[float, float], optional): 解析対象とする温度範囲 (℃)。デフォルトは (30.0, 200.0)。
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"指定されたファイルが見つかりません: {filepath}")

        self.filepath = filepath
        self.sheet_name = sheet_name
        self.sample_length_um = sample_length_mm * 1000
        self.min_segment_points = min_segment_points
        self.temp_range_c = temp_range_c
        self.time: Optional[np.ndarray] = None  # 時間データ（numpy array）
        self.temp: Optional[np.ndarray] = None  # 温度データ（numpy array）
        self.tma: Optional[np.ndarray] = None  # TMAデータ（numpy array）
        self.segments = []
        self.expansion_data = []
        self._load_data()
        self._segment_data()
        self._process_segments()

    def _load_data(self):
        """
        ファイル拡張子に基づいてExcelファイルまたはASCファイルからデータを読み込み、
        numpy arrayとして整形するプライベートメソッド。
        """
        _, ext = os.path.splitext(self.filepath)
        ext = ext.lower()

        if ext in [".xlsx", ".xls"]:
            self._load_excel_data()
        elif ext in [".asc"]:
            self._load_asc_data()
        else:
            raise ValueError(f"サポートされていないファイル形式です: {ext}")

    def _load_excel_data(self):
        """
        Excelファイルからデータを読み込み、numpy arrayに変換するプライベートメソッド。
        """
        xls = pd.ExcelFile(self.filepath)
        # 試行するシートのリストを決定。指定されたシートを最初に試す。
        sheet_names_to_try = [self.sheet_name]
        other_sheets = [
            s for s in xls.sheet_names if s != self.sheet_name and isinstance(s, str)
        ]
        sheet_names_to_try.extend(other_sheets)

        found_df = None
        found_sheet_name = ""
        header_index = -1

        for sheet_name in sheet_names_to_try:
            try:
                df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
                # 'Time(min)'を含むヘッダー行を探す
                for row_idx in range(min(50, len(df))):  # 最初の50行から検索
                    row = df.iloc[row_idx]
                    for col_idx, cell_value in enumerate(row):
                        if (
                            isinstance(cell_value, str)
                            and cell_value.strip().lower() == "time(min)"
                        ):
                            header_index = row_idx
                            found_df = df
                            found_sheet_name = sheet_name
                            break
                    if found_df is not None:
                        break
                if found_df is not None:
                    break
            except Exception:
                # シートが読み込めない、空であるなどのエラーは無視して次のシートへ
                continue

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

            required_cols = ["Time(min)", "Temp.(℃)", "TMA(μm)"]
            clean_data = (
                data_df[required_cols].apply(pd.to_numeric, errors="coerce").dropna()
            )

            # numpy arrayに変換
            self.time = clean_data["Time(min)"].to_numpy()
            self.temp = clean_data["Temp.(℃)"].to_numpy()
            self.tma = clean_data["TMA(μm)"].to_numpy()

        except KeyError:
            raise ValueError(
                f"シート '{found_sheet_name}' でヘッダーは見つかりましたが、必要な列（'Time(min)', 'Temp.(℃)', 'TMA(μm)'）が揃っていません。"
            )

    def _load_asc_data(self):
        """
        ASCファイルからデータを読み込み、numpy arrayとして整形するプライベートメソッド。
        """
        try:
            with open(self.filepath, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception as e:
            raise IOError(
                f"ファイル '{os.path.basename(self.filepath)}' の読み込みに失敗しました: {e}"
            )

        # #GDで始まる行からデータを抽出
        time_list = []
        temp_list = []
        tma_list = []

        for line in lines:
            line = line.strip()
            if line.startswith("#GD"):
                # '#GD'を除去してタブ区切りで分割
                data_part = line[4:]  # '#GD\t'を除去
                values = data_part.split("\t")
                if len(values) >= 3:
                    try:
                        time_val = float(values[0])
                        temp_val = float(values[1])
                        tma_val = float(values[2])
                        time_list.append(time_val)
                        temp_list.append(temp_val)
                        tma_list.append(tma_val)
                    except ValueError:
                        continue  # 数値に変換できない行はスキップ

        if not time_list:
            raise ValueError(
                f"ファイル '{os.path.basename(self.filepath)}' から有効なデータを読み込めませんでした。"
            )

        # 個別のnumpy arrayに変換
        self.time = np.array(time_list)
        self.temp = np.array(temp_list)
        self.tma = np.array(tma_list)

        if self.time.size == 0:
            raise ValueError(
                f"ファイル '{os.path.basename(self.filepath)}' にデータが含まれていません。"
            )

    def _segment_data(self):
        """
        温度の上昇サイクルに基づいてデータをセグメントに分割するプライベートメソッド。
        """
        if (
            self.temp is None
            or self.time is None
            or self.tma is None
            or self.temp.size == 0
            or len(self.temp) < 2
        ):
            raise ValueError("データが正常に読み込まれていません。")

        _temp = -float("inf")
        current_segment_indices = []

        for i in range(len(self.temp)):
            temp = self.temp[i]
            if temp >= _temp:
                current_segment_indices.append(i)
            else:
                if len(current_segment_indices) >= self.min_segment_points:
                    # セグメントを辞書形式で保存
                    segment_data = {
                        "time": self.time[current_segment_indices],
                        "temp": self.temp[current_segment_indices],
                        "tma": self.tma[current_segment_indices],
                    }
                    self.segments.append(segment_data)
                current_segment_indices = [i]
            _temp = temp

        # 最後のセグメントを追加
        if len(current_segment_indices) >= self.min_segment_points:
            segment_data = {
                "time": self.time[current_segment_indices],
                "temp": self.temp[current_segment_indices],
                "tma": self.tma[current_segment_indices],
            }
            self.segments.append(segment_data)

    def _process_segments(self):
        """
        各セグメントを処理し、熱膨張率を計算するプライベートメソッド。
        """
        for seg_dict in self.segments:
            temp_data = seg_dict["temp"]
            tma_data = seg_dict["tma"]

            # 温度範囲でフィルタリング
            mask = (temp_data >= self.temp_range_c[0]) & (
                temp_data <= self.temp_range_c[1]
            )
            filtered_temp = temp_data[mask]
            filtered_tma = tma_data[mask]

            if len(filtered_temp) > 0:
                initial_tma = filtered_tma[0]
                L0 = self.sample_length_um + initial_tma
                expansion = ((self.sample_length_um + filtered_tma) - L0) / L0 * 100

                # 辞書形式でexpansion_dataに追加
                expansion_dict = {"temp": filtered_temp, "expansion": expansion}
                self.expansion_data.append(expansion_dict)

    def plot_expansion(self, save_path=None, xlim=(40, 200), ylim=(0, 1.5)):
        """
        計算された熱膨張率をプロットします。
        """
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["svg.fonttype"] = "none"
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        fig, ax = plt.subplots(figsize=(5, 4))

        for i, expansion_dict in enumerate(self.expansion_data):
            temp_data = expansion_dict["temp"]
            expansion_data = expansion_dict["expansion"]
            ax.plot(temp_data, expansion_data, label=f"Segment {i+1}")

        ax.set_xlabel("Temperature(°C)")
        ax.set_ylabel("Expansion(%)")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend()
        plt.tight_layout()

        if save_path:
            # 拡張子を.svgに強制
            base, ext = os.path.splitext(save_path)
            save_path_svg = base + ".svg"
            plt.savefig(save_path_svg)
            print(f"プロットを {save_path_svg} に保存しました。")

        return fig

    def get_plot_data(self):
        """
        フロントエンド(Chart.js)に渡すためのプロット用データを生成して返す。
        """
        if not self.expansion_data:
            return []

        plot_data = []
        for i, expansion_dict in enumerate(self.expansion_data):
            temp_data = expansion_dict["temp"]
            expansion_data = expansion_dict["expansion"]
            data_points = [
                {"x": float(temp), "y": float(exp)}
                for temp, exp in zip(temp_data, expansion_data)
            ]

            series_data = {"label": f"Segment {i+1}", "data": data_points}
            plot_data.append(series_data)

        return plot_data


if __name__ == "__main__":
    # --- クラスの使用例 ---

    # 1. 解析したいファイルのパスを指定します（ExcelまたはASCファイル）
    # このパスは、ご自身のPC環境に合わせて書き換えてください。
    try:
        # Excelファイルの例
        # file_path = "C:/Users/uttya/OneDrive/神戸大学/西野研/reserch/解析データ/TMA/1_1_PMMA_lam_20250613.xlsx"

        # ASCファイルの例
        file_path = "C:/Users/uttya/OneDrive/神戸大学/西野研/04_Python/app/1_1_EP_none_20250805.ASC"

        # 2. クラスをインスタンス化します（パラメータは必要に応じて変更）
        # Excelファイルの場合、sheet_name引数でシート名を指定可能
        analyzer = TMAAnalyzer(file_path)

        # 3. プロットを作成し、ファイル名を指定して保存します
        # 元のファイル名から拡張子を除いて、新しいファイル名を生成します
        base_name = os.path.basename(file_path)
        file_name_without_ext = os.path.splitext(base_name)[0]
        output_filename = f"{file_name_without_ext}_expansion_plot.svg"

        # 保存先ディレクトリも元のコードを参考にしています
        save_dir = "C:/Users/uttya/OneDrive/神戸大学/西野研/01_報告会/fig"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  # 保存先ディレクトリがなければ作成

        output_path = os.path.join(save_dir, output_filename)

        fig = analyzer.plot_expansion(save_path=output_path)
        plt.show()  # グラフを表示
        plt.close(fig)  # メモリを解放

        print("\n処理が完了しました。")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"エラーが発生しました: {e}")
