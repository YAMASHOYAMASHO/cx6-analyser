import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.axes import Axes
from scipy.signal import savgol_filter
from scipy import interpolate
from scipy.integrate import simpson
import TMA
import DMA
import os


def create_tma_dma_figure(
    ep_temp: np.ndarray,
    ep_expansion: np.ndarray,
    peek_temp: np.ndarray,
    peek_expansion: np.ndarray,
    dma_temp: np.ndarray,
    dma_ereal: np.ndarray,
    title: str = "TMA and DMA Data",
):
    """TMA（EP/PEEK）とDMAの重ね合わせ図を作成して返す。

    Returns: (fig, ax_left, ax_right)
    - ax_left: DMA（対数）
    - ax_right: TMA（線形）
    """
    fig, ax_left = plt.subplots(figsize=(10, 6))
    ax_left.set_yscale("log", base=10)
    ax_right = ax_left.twinx()

    # TMA
    ax_right.plot(ep_temp, ep_expansion, label="EP TMA", color="blue", linewidth=2)
    ax_right.plot(
        peek_temp, peek_expansion, label="PEEK TMA", color="orange", linewidth=2
    )
    ax_right.set_xlabel("Temperature (°C)", fontsize=14)
    ax_right.set_ylabel("Thermal expansion (%)", fontsize=14)

    # DMA
    ax_left.plot(dma_temp, dma_ereal, label="DMA E'", color="green", linewidth=2)
    ax_left.set_ylabel("E' (MPa)", fontsize=14)

    l1, lab1 = ax_left.get_legend_handles_labels()
    l2, lab2 = ax_right.get_legend_handles_labels()
    ax_left.legend(l1 + l2, lab1 + lab2, loc="best", fontsize=11)
    ax_left.grid(True, alpha=0.3, which="both")
    plt.title(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    return fig, ax_left, ax_right


def plot_comparison_on_ax(
    ax: Axes,
    data_type: str,
    series: list[tuple[np.ndarray, np.ndarray]],
    labels: list[str],
    selected_index: int | None = None,
):
    """複数系列を同一Axesに比較表示するユーティリティ。

    - data_type: "TMA" or "DMA"（DMAは対数表示）
    - series: [(x, y), ...]
    - labels: 各系列のラベル
    - selected_index: 強調表示するインデックス
    """
    if data_type.upper() == "DMA":
        ax.set_yscale("log")

    for i, ((x, y), label) in enumerate(zip(series, labels)):
        is_sel = selected_index is not None and i == selected_index
        color = "red" if is_sel else f"C{i}"
        lw = 3 if is_sel else 1.5
        alpha = 1.0 if is_sel else 0.6
        z = 10 if is_sel else 1
        ax.plot(x, y, color=color, linewidth=lw, alpha=alpha, label=label, zorder=z)

    if data_type.upper() == "TMA":
        ax.set_ylabel("Thermal expansion (%)", fontsize=12)
        ax.set_title("TMA Data Comparison", fontsize=14, fontweight="bold")
    else:
        ax.set_ylabel("E' (MPa)", fontsize=12)
        ax.set_title("DMA Data Comparison", fontsize=14, fontweight="bold")

    ax.set_xlabel("Temperature (°C)", fontsize=12)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="best")


class FileSelector:
    """TMAとDMAファイルを選択し、可視化するクラス"""

    def __init__(self, data_type="TMA"):
        """
        Parameters:
        -----------
        data_type : str
            'TMA' または 'DMA' を指定
        """
        self.data_type = data_type
        self.file_paths = []
        self.data_objects = []
        self.selected_file_path = None
        self.root = None

    def select_files(self):
        """ファイル選択ダイアログを表示"""
        title = f"Select {self.data_type} data file(s)"
        self.file_paths = list(
            filedialog.askopenfilenames(title=title, filetypes=[("All files", "*.*")])
        )

        if not self.file_paths:
            return None

        # データオブジェクトの作成（失敗時は該当ファイルを除外）
        valid_paths = []
        valid_objects = []
        for path in self.file_paths:
            try:
                if self.data_type == "TMA":
                    obj = TMA.TMAAnalyzer(path)
                elif self.data_type == "DMA":
                    obj = DMA.DMAAnalyzer(path)
                else:
                    continue
                valid_paths.append(path)
                valid_objects.append(obj)
            except Exception as e:
                print(f"Error loading {self.data_type} file {path}: {e}")
        self.file_paths = valid_paths
        self.data_objects = valid_objects
        return self.file_paths

    def visualize_and_select(self):
        """選択したファイルを可視化し、使用するファイルを決定

        備考:
        - 選択されたファイルが1つだけの場合はプレビューをスキップして即時確定する。
        """
        if not self.file_paths:
            print("No files selected.")
            return None

        # ファイルが1つのみの場合は確認をスキップして即時確定
        if len(self.file_paths) == 1:
            self.selected_file_path = self.file_paths[0]
            return self.selected_file_path

        self.root = tk.Tk()
        self.root.title(f"{self.data_type} File Selector")
        self.root.geometry("1200x700")

        # ×ボタンを無効化（Confirm Selectionボタンでのみ閉じられる）
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # 左側:ラジオボタンフレーム
        left_frame = ttk.Frame(self.root)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)

        ttk.Label(
            left_frame, text="Select file to use:", font=("Arial", 12, "bold")
        ).pack(anchor=tk.W, pady=5)

        self.selected_var = tk.StringVar(value=self.file_paths[0])

        # ラジオボタンの作成（縦に配置）
        radio_frame = ttk.Frame(left_frame)
        radio_frame.pack(fill=tk.BOTH, expand=True)

        for path in self.file_paths:
            filename = os.path.basename(path)
            ttk.Radiobutton(
                radio_frame,
                text=filename,
                variable=self.selected_var,
                value=path,
                command=self._update_plot,
            ).pack(anchor=tk.W, pady=2)

        # 確定ボタン
        ttk.Button(
            left_frame, text="Confirm Selection", command=self._confirm_selection
        ).pack(pady=10)

        # 右側：プロットフレーム
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Matplotlibの設定
        plt.rcParams["figure.figsize"] = [8, 6]
        plt.rcParams["font.size"] = 10

        # 全データを重ねてプロット
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 凡例用のリスト
        self.line_objects = {}

        # 初期プロット
        self._update_plot()

        self.root.mainloop()

        return self.selected_file_path

    def _update_plot(self):
        """選択されたファイルのプロットを更新（全データ表示、選択されたものを赤く）"""
        self.ax.clear()
        self.line_objects.clear()

        selected_path = self.selected_var.get()

        if self.data_type == "TMA":
            # 全TMAデータをプロット
            for i, (path, data_obj) in enumerate(
                zip(self.file_paths, self.data_objects)
            ):
                segment = data_obj.segments[-1]
                temp = segment["temp"]
                tma = segment["tma"]
                expansion = savgol_filter(tma, 51, 3)
                expansion = (expansion - expansion[0]) / (expansion[0] + 20000) * 100

                filename = os.path.basename(path)
                is_selected = path == selected_path
                color = "red" if is_selected else f"C{i}"
                linewidth = 3 if is_selected else 1.5
                alpha = 1.0 if is_selected else 0.5
                zorder = 10 if is_selected else 1

                (line,) = self.ax.plot(
                    temp,
                    expansion,
                    linewidth=linewidth,
                    color=color,
                    alpha=alpha,
                    label=filename,
                    zorder=zorder,
                )
                self.line_objects[path] = line

            self.ax.set_xlabel("Temperature (°C)", fontsize=12)
            self.ax.set_ylabel("Thermal expansion (%)", fontsize=12)
            self.ax.set_title("TMA Data Comparison", fontsize=14, fontweight="bold")
            self.ax.grid(True, alpha=0.3)
            self.ax.legend(fontsize=9, loc="best")

        elif self.data_type == "DMA":
            # 全DMAデータをプロット
            self.ax.set_yscale("log")

            for i, (path, data_obj) in enumerate(
                zip(self.file_paths, self.data_objects)
            ):
                temp = data_obj.data["temp"]
                ereal = data_obj.data["Ereal"]

                filename = os.path.basename(path)
                is_selected = path == selected_path
                color = "red" if is_selected else f"C{i}"
                linewidth = 3 if is_selected else 1.5
                alpha = 1.0 if is_selected else 0.5
                zorder = 10 if is_selected else 1

                (line,) = self.ax.plot(
                    temp,
                    ereal,
                    linewidth=linewidth,
                    color=color,
                    alpha=alpha,
                    label=filename,
                    zorder=zorder,
                )
                self.line_objects[path] = line

            self.ax.set_xlabel("Temperature (°C)", fontsize=12)
            self.ax.set_ylabel("E' (MPa)", fontsize=12)
            self.ax.set_title("DMA Data Comparison", fontsize=14, fontweight="bold")
            self.ax.grid(True, alpha=0.3, which="both")
            self.ax.legend(fontsize=9, loc="best")

        self.fig.tight_layout()
        self.canvas.draw()

    def _on_closing(self):
        """×ボタンが押された時の処理（何もしない）"""
        # メッセージを表示して無視
        pass

    def _confirm_selection(self):
        """選択を確定してウィンドウを閉じる"""
        self.selected_file_path = self.selected_var.get()
        if self.root is not None:
            self.root.quit()  # mainloopを終了
            self.root.destroy()  # ウィンドウを破棄


def calculate_modified_residual_stress(
    ep_tma_path,
    peek_tma_path,
    ep_dma_path,
    peek_dma_path,
    init_temp=40,
    final_temp=200,
    ep_poisson_ratio=0.4,
    peek_poisson_ratio=0.4,
    save_figures=False,
):
    """
    修正式を用いて残留応力を計算し、結果を可視化・出力する関数

    修正式: E_modified = (E_PEEK × E_EP) / (E_PEEK - E_EP)
    被積分関数: E_modified / (1-ν) × (α_PEEK - α_EP)

    Parameters:
    -----------
    ep_tma_path : str
        EP（フィルム）のTMAデータファイルパス
    peek_tma_path : str
        PEEK（基板）のTMAデータファイルパス
    ep_dma_path : str
        EP（フィルム）のDMAデータファイルパス
    peek_dma_path : str
        PEEK（基板）のDMAデータファイルパス
    init_temp : float
        計算開始温度 (°C)
    final_temp : float
        計算終了温度 (°C)
    ep_poisson_ratio : float
        EP（フィルム）のポアソン比
    peek_poisson_ratio : float
        PEEK（基板）のポアソン比
    save_figures : bool
        Trueの場合、グラフを保存

    Returns:
    --------
    dict
        計算結果を含む辞書
        - 'residual_stress': 残留応力値 (MPa)
        - 'temp_range': 温度範囲
        - 'stress_vs_temp': 各温度での残留応力
    """

    # データの読み込み
    ep_tma_data = TMA.TMAAnalyzer(ep_tma_path)
    peek_tma_data = TMA.TMAAnalyzer(peek_tma_path)
    ep_dma_data = DMA.DMAAnalyzer(ep_dma_path)
    peek_dma_data = DMA.DMAAnalyzer(peek_dma_path)

    # TMAデータの処理（EP：フィルム）
    ep_tma_temp = ep_tma_data.segments[-1]["temp"]
    ep_tma_expansion = savgol_filter(ep_tma_data.segments[-1]["tma"], 51, 3)
    ep_tma_expansion = (
        (ep_tma_expansion - ep_tma_expansion[0]) / (ep_tma_expansion[0] + 20000) * 100
    )
    final_ep_tma_expansion = ep_tma_expansion[
        (ep_tma_temp >= init_temp) & (ep_tma_temp <= final_temp)
    ]
    final_ep_tma_temp = ep_tma_temp[
        (ep_tma_temp >= init_temp) & (ep_tma_temp <= final_temp)
    ]

    # TMAデータの処理（PEEK：基板）
    peek_tma_temp = peek_tma_data.segments[-1]["temp"]
    peek_tma_expansion = savgol_filter(peek_tma_data.segments[-1]["tma"], 51, 3)
    peek_tma_expansion = (
        (peek_tma_expansion - peek_tma_expansion[0])
        / (peek_tma_expansion[0] + 20000)
        * 100
    )
    final_peek_tma_expansion = peek_tma_expansion[
        (peek_tma_temp >= init_temp) & (peek_tma_temp <= final_temp)
    ]
    final_peek_tma_temp = peek_tma_temp[
        (peek_tma_temp >= init_temp) & (peek_tma_temp <= final_temp)
    ]

    # DMAデータの処理（EP）
    ep_dma_temp = ep_dma_data.data["temp"]
    ep_dma_ereal = savgol_filter(ep_dma_data.data["Ereal"], 51, 3)
    final_ep_dma_ereal = ep_dma_ereal[
        (ep_dma_temp >= init_temp) & (ep_dma_temp <= final_temp)
    ]
    final_ep_dma_temp = ep_dma_temp[
        (ep_dma_temp >= init_temp) & (ep_dma_temp <= final_temp)
    ]

    # DMAデータの処理（PEEK）
    peek_dma_temp = peek_dma_data.data["temp"]
    peek_dma_ereal = savgol_filter(peek_dma_data.data["Ereal"], 51, 3)
    final_peek_dma_ereal = peek_dma_ereal[
        (peek_dma_temp >= init_temp) & (peek_dma_temp <= final_temp)
    ]
    final_peek_dma_temp = peek_dma_temp[
        (peek_dma_temp >= init_temp) & (peek_dma_temp <= final_temp)
    ]

    # 共通の温度軸を作成
    temp_range = np.arange(init_temp, final_temp, 0.1)

    # 補間（TMA）
    ep_tma_interp = interpolate.interp1d(
        final_ep_tma_temp,
        final_ep_tma_expansion,
        kind="cubic",
        fill_value="extrapolate",
    )
    ep_tma_interp_vals = ep_tma_interp(temp_range)
    ep_alpha = np.gradient(ep_tma_interp_vals, temp_range) * 1e-2

    peek_tma_interp = interpolate.interp1d(
        final_peek_tma_temp,
        final_peek_tma_expansion,
        kind="cubic",
        fill_value="extrapolate",
    )
    peek_tma_interp_vals = peek_tma_interp(temp_range)
    peek_alpha = np.gradient(peek_tma_interp_vals, temp_range) * 1e-2

    # 補間（DMA）
    ep_e_interp = interpolate.interp1d(
        final_ep_dma_temp, final_ep_dma_ereal, kind="cubic", fill_value="extrapolate"
    )
    ep_e_vals = ep_e_interp(temp_range)

    peek_e_interp = interpolate.interp1d(
        final_peek_dma_temp,
        final_peek_dma_ereal,
        kind="cubic",
        fill_value="extrapolate",
    )
    peek_e_vals = peek_e_interp(temp_range)

    # 修正式によるヤング率の計算
    # E_modified = (E_PEEK/(1-ν_PEEK) × E_EP/(1-ν_EP)) / (E_PEEK/(1-ν_PEEK) - E_EP/(1-ν_EP))
    # ゼロ除算を避けるため、差が小さい場合は平均値を使用

    # それぞれのEに1/(1-ν)を適用
    ep_e_modified = ep_e_vals / (1 - ep_poisson_ratio)
    peek_e_modified = peek_e_vals / (1 - peek_poisson_ratio)

    e_diff = peek_e_modified + ep_e_modified
    e_modified = np.where(
        np.abs(e_diff) > 1e-6,  # 差が十分大きい場合
        (peek_e_modified * ep_e_modified) / e_diff,
        (peek_e_modified + ep_e_modified) / 2,  # 差が小さい場合は平均値
    )

    # 平均ポアソン比（簡易的に体積比で重み付け）
    # ここでは単純平均を使用
    avg_poisson = (ep_poisson_ratio + peek_poisson_ratio) / 2

    # 被積分関数の計算（修正式）
    # E_modified × (α_PEEK - α_EP)
    integrand = e_modified * (peek_alpha - ep_alpha) * 1e-6

    # 残留応力の計算
    residual_stress = simpson(integrand, temp_range)

    # 各温度での残留応力を計算
    stress_vs_temp = []
    temp_points = np.arange(init_temp, final_temp + 1, 1)
    for t in temp_points:
        mask = (temp_range >= init_temp) & (temp_range <= t)
        stress = simpson(integrand[mask], temp_range[mask])
        stress_vs_temp.append(stress)

    # Matplotlibの設定
    plt.rcParams["figure.figsize"] = [8, 6]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.size"] = 12

    # グラフ1: TMAとDMAの重ね合わせ（EP）
    fig1, ax1_dma = plt.subplots(figsize=(10, 6))
    ax1_dma.set_yscale("log", base=10)
    ax1_tma = ax1_dma.twinx()

    ax1_tma.plot(
        final_ep_tma_temp,
        final_ep_tma_expansion,
        label="EP TMA",
        color="blue",
        linewidth=2,
    )
    ax1_tma.set_xlabel("Temperature (°C)", fontsize=14)
    ax1_tma.set_ylabel("Thermal expansion (%)", fontsize=14, color="blue")

    ax1_dma.plot(
        final_ep_dma_temp,
        final_ep_dma_ereal,
        label="EP DMA E'",
        color="green",
        linewidth=2,
    )
    ax1_dma.set_ylabel("E' (MPa)", fontsize=14, color="green")

    lines1, labels1 = ax1_dma.get_legend_handles_labels()
    lines2, labels2 = ax1_tma.get_legend_handles_labels()
    ax1_dma.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=11)
    ax1_dma.grid(True, alpha=0.3, which="both")
    plt.title("EP: TMA and DMA Data", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # グラフ2: TMAとDMAの重ね合わせ（PEEK）
    fig2, ax2_dma = plt.subplots(figsize=(10, 6))
    ax2_dma.set_yscale("log", base=10)
    ax2_tma = ax2_dma.twinx()

    ax2_tma.plot(
        final_peek_tma_temp,
        final_peek_tma_expansion,
        label="PEEK TMA",
        color="orange",
        linewidth=2,
    )
    ax2_tma.set_xlabel("Temperature (°C)", fontsize=14)
    ax2_tma.set_ylabel("Thermal expansion (%)", fontsize=14, color="orange")

    ax2_dma.plot(
        final_peek_dma_temp,
        final_peek_dma_ereal,
        label="PEEK DMA E'",
        color="purple",
        linewidth=2,
    )
    ax2_dma.set_ylabel("E' (MPa)", fontsize=14, color="purple")

    lines1, labels1 = ax2_dma.get_legend_handles_labels()
    lines2, labels2 = ax2_tma.get_legend_handles_labels()
    ax2_dma.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=11)
    ax2_dma.grid(True, alpha=0.3, which="both")
    plt.title("PEEK: TMA and DMA Data", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # グラフ3: DMAの比較（EP vs PEEK）
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.set_yscale("log", base=10)
    ax3.plot(
        final_ep_dma_temp,
        final_ep_dma_ereal,
        label="EP E'",
        color="blue",
        linewidth=2,
    )
    ax3.plot(
        final_peek_dma_temp,
        final_peek_dma_ereal,
        label="PEEK E'",
        color="orange",
        linewidth=2,
    )
    ax3.plot(
        temp_range,
        e_modified,
        label="E_modified",
        color="red",
        linewidth=2,
        linestyle="--",
    )
    ax3.set_xlabel("Temperature (°C)", fontsize=14)
    ax3.set_ylabel("E' (MPa)", fontsize=14)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3, which="both")
    plt.title("DMA Comparison and Modified E", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # グラフ4: 線熱膨張係数
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.plot(
        temp_range,
        peek_alpha * 1e6,
        label="PEEK α (Substrate)",
        color="blue",
        linewidth=2,
    )
    ax4.plot(
        temp_range, ep_alpha * 1e6, label="EP α (Film)", color="orange", linewidth=2
    )
    ax4.set_xlabel("Temperature (°C)", fontsize=14)
    ax4.set_ylabel("Thermal expansion coefficient (×10⁻⁶ /°C)", fontsize=14)
    ax4.legend(fontsize=12)
    ax4.grid(True, alpha=0.3)
    plt.title("Thermal Expansion Coefficients", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # グラフ5: 残留応力の温度依存性
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    ax5.scatter(temp_points, stress_vs_temp, color="red", s=20, alpha=0.6)
    ax5.plot(temp_points, stress_vs_temp, color="red", linewidth=2, alpha=0.8)
    ax5.set_xlabel("Temperature (°C)", fontsize=14)
    ax5.set_ylabel("Residual Stress (MPa)", fontsize=14)
    ax5.grid(True, alpha=0.3)
    plt.title(
        "Residual Stress vs Temperature (Modified Formula)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    # 結果テキストの作成
    result_text = []
    result_text.append("=" * 70)
    result_text.append("MODIFIED RESIDUAL STRESS CALCULATION RESULTS")
    result_text.append("=" * 70)
    result_text.append(f"Final Residual Stress: {residual_stress:.2f} MPa")
    result_text.append("")
    result_text.append("Modified Formula:")
    result_text.append("  E_modified = (E_PEEK × E_EP) / (E_PEEK - E_EP)")
    result_text.append("  Integrand = E_modified / (1-ν) × (α_PEEK - α_EP)")
    result_text.append("")
    result_text.append("Calculation Conditions:")
    result_text.append(f"  - Temperature range: {init_temp}°C to {final_temp}°C")
    result_text.append(f"  - EP Poisson's ratio: {ep_poisson_ratio}")
    result_text.append(f"  - PEEK Poisson's ratio: {peek_poisson_ratio}")
    result_text.append(f"  - Average Poisson's ratio (used): {avg_poisson:.3f}")
    result_text.append(f"  - Average EP Young's modulus: {np.mean(ep_e_vals):.2f} MPa")
    result_text.append(
        f"  - Average PEEK Young's modulus: {np.mean(peek_e_vals):.2f} MPa"
    )
    result_text.append(
        f"  - Average Modified Young's modulus: {np.mean(e_modified):.2f} MPa"
    )
    result_text.append(
        f"  - Average thermal expansion coefficient difference (αPEEK-αEP): {np.mean(peek_alpha - ep_alpha):.6e} 1/°C"
    )
    result_text.append("")
    result_text.append("Input Files:")
    result_text.append(f"  - EP (Film) TMA: {os.path.basename(ep_tma_path)}")
    result_text.append(f"  - PEEK (Substrate) TMA: {os.path.basename(peek_tma_path)}")
    result_text.append(f"  - EP (Film) DMA: {os.path.basename(ep_dma_path)}")
    result_text.append(f"  - PEEK (Substrate) DMA: {os.path.basename(peek_dma_path)}")
    result_text.append("=" * 70)

    # グラフとテキストの保存
    if save_figures:
        save_dir = filedialog.askdirectory(title="Select directory to save figures")
        if save_dir:
            # グラフの保存
            fig1.savefig(
                os.path.join(save_dir, "ep_tma_dma_data.svg"),
                format="svg",
                bbox_inches="tight",
            )
            fig2.savefig(
                os.path.join(save_dir, "peek_tma_dma_data.svg"),
                format="svg",
                bbox_inches="tight",
            )
            fig3.savefig(
                os.path.join(save_dir, "dma_comparison_modified_e.svg"),
                format="svg",
                bbox_inches="tight",
            )
            fig4.savefig(
                os.path.join(save_dir, "thermal_expansion_coefficients.svg"),
                format="svg",
                bbox_inches="tight",
            )
            fig5.savefig(
                os.path.join(save_dir, "residual_stress_vs_temp_modified.svg"),
                format="svg",
                bbox_inches="tight",
            )

            # テキストファイルの保存
            result_file_path = os.path.join(
                save_dir, "calculation_results_modified.txt"
            )
            with open(result_file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(result_text))

            print(f"\nFigures and results saved to: {save_dir}")
            print(f"  - ep_tma_dma_data.svg")
            print(f"  - peek_tma_dma_data.svg")
            print(f"  - dma_comparison_modified_e.svg")
            print(f"  - thermal_expansion_coefficients.svg")
            print(f"  - residual_stress_vs_temp_modified.svg")
            print(f"  - calculation_results_modified.txt")

    plt.show()

    # 結果の表示
    print("\n" + "\n".join(result_text) + "\n")

    return {
        "residual_stress": residual_stress,
        "temp_range": temp_points,
        "stress_vs_temp": np.array(stress_vs_temp),
        "ep_alpha": ep_alpha,
        "peek_alpha": peek_alpha,
        "ep_youngs_modulus": ep_e_vals,
        "peek_youngs_modulus": peek_e_vals,
        "modified_youngs_modulus": e_modified,
    }


# 使用例
if __name__ == "__main__":
    # ステップ1: EP TMAファイルの選択
    print("Step 1: Select EP (Film) TMA file")
    ep_tma_selector = FileSelector("TMA")
    ep_tma_selector.select_files()
    ep_tma_path = ep_tma_selector.visualize_and_select()

    # ステップ2: PEEK TMAファイルの選択
    print("\nStep 2: Select PEEK (Substrate) TMA file")
    peek_tma_selector = FileSelector("TMA")
    peek_tma_selector.select_files()
    peek_tma_path = peek_tma_selector.visualize_and_select()

    # ステップ3: EP DMAファイルの選択
    print("\nStep 3: Select EP (Film) DMA file")
    ep_dma_selector = FileSelector("DMA")
    ep_dma_selector.select_files()
    ep_dma_path = ep_dma_selector.visualize_and_select()

    # ステップ4: PEEK DMAファイルの選択
    print("\nStep 4: Select PEEK (Substrate) DMA file")
    peek_dma_selector = FileSelector("DMA")
    peek_dma_selector.select_files()
    peek_dma_path = peek_dma_selector.visualize_and_select()

    # ステップ5: 計算条件入力フォーム
    from tkinter import simpledialog

    class ModifiedParamForm(simpledialog.Dialog):
        def body(self, master):
            tk.Label(master, text="初期温度 (°C)").grid(row=0, column=0, sticky=tk.W)
            tk.Label(master, text="最終温度 (°C)").grid(row=1, column=0, sticky=tk.W)
            tk.Label(master, text="EP ポアソン比").grid(row=2, column=0, sticky=tk.W)
            tk.Label(master, text="PEEK ポアソン比").grid(row=3, column=0, sticky=tk.W)

            self.e1 = tk.Entry(master)
            self.e2 = tk.Entry(master)
            self.e3 = tk.Entry(master)
            self.e4 = tk.Entry(master)

            self.e1.insert(0, "40")
            self.e2.insert(0, "200")
            self.e3.insert(0, "0.4")
            self.e4.insert(0, "0.4")

            self.e1.grid(row=0, column=1)
            self.e2.grid(row=1, column=1)
            self.e3.grid(row=2, column=1)
            self.e4.grid(row=3, column=1)

            return self.e1

        def apply(self):
            self.result = {
                "init_temp": float(self.e1.get()),
                "final_temp": float(self.e2.get()),
                "ep_poisson_ratio": float(self.e3.get()),
                "peek_poisson_ratio": float(self.e4.get()),
            }

    root = tk.Tk()
    root.withdraw()
    param_dialog = ModifiedParamForm(root, title="計算条件の入力（修正式）")
    params = (
        param_dialog.result
        if param_dialog.result
        else {
            "init_temp": 40,
            "final_temp": 200,
            "ep_poisson_ratio": 0.4,
            "peek_poisson_ratio": 0.4,
        }
    )
    root.destroy()

    # ステップ6: 計算実行
    if ep_tma_path and peek_tma_path and ep_dma_path and peek_dma_path:
        print("\nStep 5: Calculate modified residual stress")
        results = calculate_modified_residual_stress(
            ep_tma_path=ep_tma_path,
            peek_tma_path=peek_tma_path,
            ep_dma_path=ep_dma_path,
            peek_dma_path=peek_dma_path,
            init_temp=params["init_temp"],
            final_temp=params["final_temp"],
            ep_poisson_ratio=params["ep_poisson_ratio"],
            peek_poisson_ratio=params["peek_poisson_ratio"],
            save_figures=True,
        )
    else:
        print("File selection was cancelled.")
