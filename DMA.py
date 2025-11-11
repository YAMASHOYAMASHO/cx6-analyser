"""
このモジュールは、DMA（動的粘弾性測定）データのテキストファイルを解析し、
物性値のプロットやCSV保存を行うためのクラスを提供します。

主な機能:
    - DMA測定データの読み込みと解析
    - E', E''、tanδのプロット作成
    - 解析結果のCSV保存
    - データの異常値チェック
    - Chart.js用データ生成

主なクラス:
    DMAAnalyzer: DMAデータの解析・可視化・保存を行うクラス

使用方法:
    DMAAnalyzerクラスをインスタンス化し、plot_dmaやsave_csvなどのメソッドを利用します。
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

class DMAAnalyzer:
    """
    DMA（動的粘弾性測定）のテキストファイルを解析し、
    プロット作成とCSV保存を行うためのクラス。
    """

    def __init__(self, filepath, header_rows=12, footer_rows=1, encoding='ANSI'):
        """
        DMAAnalyzerのインスタンスを初期化します。

        Args:
            filepath (str): 解析対象のテキストファイルのパス。
            header_rows (int, optional): スキップするヘッダーの行数。デフォルトは12。
            footer_rows (int, optional): スキップするフッターの行数。デフォルトは1。
            encoding (str, optional): ファイルのエンコーディング。デフォルトは'ANSI'。
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"指定されたファイルが見つかりません: {filepath}")

        self.filepath = filepath
        self.name = os.path.splitext(os.path.basename(filepath))[0]
        self.header_rows = header_rows
        self.footer_rows = footer_rows
        self.encoding = encoding
        self.data = None
        self._load_and_parse_data()

        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["svg.fonttype"] = "none"
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"

    def _load_and_parse_data(self):
        """
        テキストファイルを読み込み、DataFrameに変換するプライベートメソッド。
        """
        with open(self.filepath, 'r', encoding=self.encoding) as f:
            lines = f.readlines()
        
        # ヘッダーとフッターを除外
        data_lines = lines[self.header_rows : -self.footer_rows if self.footer_rows > 0 else None]
        
        # データを分割し、floatに変換
        parsed_data = [list(map(float, line.split())) for line in data_lines]
        
        if '動的粘弾性測定' in lines[0]:
            columns = ['temp', 'Ereal', 'Eimag', 'tanδ', 'SL', 'SF', 'DF', 'DD']
        else:
            columns = ['time', 'temp', 'Ereal', 'Eimag', 'tanδ', 'SF', 'DS', 'SL']
        self.data = pd.DataFrame(parsed_data, columns=columns)

    def plot_dma(self, save_path=None, xlim=(30, 200), ylim1=(1e0, 1e10), ylim2=(0, 0.2)):
        """
        DMAの物性値（E', E'', tanδ）をプロットします。

        Args:
            save_path (str, optional): プロットを保存するパス。指定しない場合、保存されない。
            xlim (tuple, optional): X軸の表示範囲。
            ylim1 (tuple, optional): 主軸（E', E''）のY軸表示範囲。
            ylim2 (tuple, optional): 第二軸（tanδ）のY軸表示範囲。
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。ファイルの形式を確認してください。")
            
        fig, ax1 = plt.subplots(figsize=(5, 4))

        # 主軸 (E', E'')
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel("E', E'' (Pa)")
        ax1.set_yscale('log')
        p1, = ax1.plot(self.data['temp'], self.data['Ereal'], 'b-', label="E' (Ereal)")
        p2, = ax1.plot(self.data['temp'], self.data['Eimag'], 'g-', label="E'' (Eimag)")
        ax1.set_ylim(ylim1)
        ax1.set_xlim(xlim)

        # 第二軸 (tanδ)
        ax2 = ax1.twinx()
        ax2.set_ylabel('tanδ')
        p3, = ax2.plot(self.data['temp'], self.data['tanδ'], 'r-', label='tanδ')
        ax2.set_ylim(ylim2)
        
        # 凡例をまとめる
        lines = [p1, p2, p3]
        ax1.legend(lines, [l.get_label() for l in lines], loc='upper left')

        fig.tight_layout()
        
        if save_path:
            # 拡張子を.svgに強制
            base, ext = os.path.splitext(save_path)
            save_path_svg = base + ".svg"
            plt.savefig(save_path_svg)
            print(f"プロットを {save_path_svg} に保存しました。")
        
    def save_csv(self, save_path=None):
        """
        解析されたデータをCSVファイルとして保存します。

        Args:
            save_path (str, optional): CSVを保存するパス。指定しない場合、
                                     入力ファイル名に基づいた名前で保存される。
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。ファイルの形式を確認してください。")
            
        if save_path is None:
            save_path = f"{self.name}_processed.csv"
            
        self.data.to_csv(save_path, index=False)
        print(f"処理済みデータを {save_path} に保存しました。")

    def get_plot_data(self):
        """
        フロントエンド(Chart.js)に渡すためのプロット用データを生成して返す。
        """
        if self.data is None:
            return {}

        return {
            'Ereal': self.data.rename(columns={'temp': 'x', 'Ereal': 'y'})[['x', 'y']].to_dict('records'),
            'Eimag': self.data.rename(columns={'temp': 'x', 'Eimag': 'y'})[['x', 'y']].to_dict('records'),
            'tanDelta': self.data.rename(columns={'temp': 'x', 'tanδ': 'y'})[['x', 'y']].to_dict('records')
        }

    def check(self):
        """
        SFとDFの値をチェックし、異常があるかどうかを確認します。
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。ファイルの形式を確認してください。")
        
        sf = self.data['SF']
        df = self.data['DF']
        for i in range(len(sf)):
            if sf[i] < df[i]:
                print(f"異常値検出: SF({sf[i]}) < DF({df[i]}) at index {i}")