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

        # データオブジェクトの作成
        self.data_objects = []
        for path in self.file_paths:
            if self.data_type == "TMA":
                self.data_objects.append(TMA.TMAAnalyzer(path))
            elif self.data_type == "DMA":
                self.data_objects.append(DMA.DMAAnalyzer(path))

        return self.file_paths

    def visualize_and_select(self):
        """選択したファイルを可視化し、使用するファイルを決定"""
        if not self.file_paths:
            print("No files selected.")
            return None

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
        self.root.quit()  # mainloopを終了
        self.root.destroy()  # ウィンドウを破棄


def calculate_residual_stress(
    ep_tma_path,
    peek_tma_path,
    dma_path,
    init_temp=40,
    final_temp=200,
    poisson_ratio=0.4,
    save_figures=False,
):
    """
    残留応力を計算し、結果を可視化・出力する関数

    Parameters:
    -----------
    ep_tma_path : str
        EP（フィルム）のTMAデータファイルパス
    peek_tma_path : str
        PEEK（基板）のTMAデータファイルパス
    dma_path : str
        DMAデータファイルパス
    init_temp : float
        計算開始温度 (°C)
    final_temp : float
        計算終了温度 (°C)
    poisson_ratio : float
        ポアソン比
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
    dma_data = DMA.DMAAnalyzer(dma_path)

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

    # DMAデータの処理
    dma_temp = dma_data.data["temp"]
    dma_ereal = savgol_filter(dma_data.data["Ereal"], 51, 3)
    final_dma_ereal = dma_ereal[(dma_temp >= init_temp) & (dma_temp <= final_temp)]
    final_dma_temp = dma_temp[(dma_temp >= init_temp) & (dma_temp <= final_temp)]

    # 共通の温度軸を作成
    temp_range = np.arange(init_temp, final_temp, 0.1)

    # 補間
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

    e_interp = interpolate.interp1d(
        final_dma_temp, final_dma_ereal, kind="cubic", fill_value="extrapolate"
    )
    e_vals = e_interp(temp_range)

    # 被積分関数の計算（基板 - フィルム = PEEK - EP）
    integrand = (e_vals / (1 - poisson_ratio)) * (peek_alpha - ep_alpha) * 1e-6

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

    # グラフ1: TMAとDMAの重ね合わせ
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_yscale("log", base=10)
    ax1_twin = ax1.twinx()

    ax1_twin.plot(
        final_ep_tma_temp,
        final_ep_tma_expansion,
        label="EP TMA",
        color="blue",
        linewidth=2,
    )
    ax1_twin.plot(
        final_peek_tma_temp,
        final_peek_tma_expansion,
        label="PEEK TMA",
        color="orange",
        linewidth=2,
    )
    ax1_twin.set_xlabel("Temperature (°C)", fontsize=14)
    ax1_twin.set_ylabel("Thermal expansion (%)", fontsize=14)

    ax1.plot(
        final_dma_temp, final_dma_ereal, label="DMA E'", color="green", linewidth=2
    )
    ax1.set_ylabel("E' (MPa)", fontsize=14)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=11)
    ax1.grid(True, alpha=0.3)
    plt.title("TMA and DMA Data", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # グラフ2: 線熱膨張係数
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(
        temp_range,
        peek_alpha * 1e6,
        label="PEEK α (Substrate)",
        color="blue",
        linewidth=2,
    )
    ax2.plot(
        temp_range, ep_alpha * 1e6, label="EP α (Film)", color="orange", linewidth=2
    )
    ax2.set_xlabel("Temperature (°C)", fontsize=14)
    ax2.set_ylabel("Thermal expansion coefficient (×10⁻⁶ /°C)", fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    plt.title("Thermal Expansion Coefficients", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # グラフ3: 残留応力の温度依存性
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.scatter(temp_points, stress_vs_temp, color="red", s=20, alpha=0.6)
    ax3.plot(temp_points, stress_vs_temp, color="red", linewidth=2, alpha=0.8)
    ax3.set_xlabel("Temperature (°C)", fontsize=14)
    ax3.set_ylabel("Residual Stress (MPa)", fontsize=14)
    ax3.grid(True, alpha=0.3)
    plt.title("Residual Stress vs Temperature", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # 結果テキストの作成
    result_text = []
    result_text.append("=" * 60)
    result_text.append("RESIDUAL STRESS CALCULATION RESULTS")
    result_text.append("=" * 60)
    result_text.append(f"Final Residual Stress: {residual_stress:.2f} MPa")
    result_text.append("")
    result_text.append("Calculation Conditions:")
    result_text.append(f"  - Temperature range: {init_temp}°C to {final_temp}°C")
    result_text.append(f"  - Poisson's ratio: {poisson_ratio}")
    result_text.append(f"  - Average Young's modulus: {np.mean(e_vals):.2f} MPa")
    result_text.append(
        f"  - Average thermal expansion coefficient difference (αPEEK-αEP): {np.mean(peek_alpha - ep_alpha):.6e} 1/°C"
    )
    result_text.append("")
    result_text.append("Input Files:")
    result_text.append(f"  - EP (Film) TMA: {os.path.basename(ep_tma_path)}")
    result_text.append(f"  - PEEK (Substrate) TMA: {os.path.basename(peek_tma_path)}")
    result_text.append(f"  - DMA: {os.path.basename(dma_path)}")
    result_text.append("=" * 60)

    # グラフとテキストの保存
    if save_figures:
        save_dir = filedialog.askdirectory(title="Select directory to save figures")
        if save_dir:
            # グラフの保存
            fig1.savefig(
                os.path.join(save_dir, "tma_dma_data.svg"),
                format="svg",
                bbox_inches="tight",
            )
            fig2.savefig(
                os.path.join(save_dir, "thermal_expansion_coefficients.svg"),
                format="svg",
                bbox_inches="tight",
            )
            fig3.savefig(
                os.path.join(save_dir, "residual_stress_vs_temp.svg"),
                format="svg",
                bbox_inches="tight",
            )

            # テキストファイルの保存
            result_file_path = os.path.join(save_dir, "calculation_results.txt")
            with open(result_file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(result_text))

            print(f"\nFigures and results saved to: {save_dir}")
            print(f"  - tma_dma_data.svg")
            print(f"  - thermal_expansion_coefficients.svg")
            print(f"  - residual_stress_vs_temp.svg")
            print(f"  - calculation_results.txt")

    plt.show()

    # 結果の表示
    print("\n" + "\n".join(result_text) + "\n")

    return {
        "residual_stress": residual_stress,
        "temp_range": temp_points,
        "stress_vs_temp": np.array(stress_vs_temp),
        "ep_alpha": ep_alpha,
        "peek_alpha": peek_alpha,
        "youngs_modulus": e_vals,
    }


# 使用例
if __name__ == "__main__":
    # ステップ1: ファイルの選択と可視化
    print("Step 1: Select EP (Film) TMA file")
    ep_selector = FileSelector("TMA")
    ep_selector.select_files()
    ep_tma_path = ep_selector.visualize_and_select()

    print("\nStep 2: Select PEEK (Substrate) TMA file")
    peek_selector = FileSelector("TMA")
    peek_selector.select_files()
    peek_tma_path = peek_selector.visualize_and_select()

    print("\nStep 3: Select DMA file")
    dma_selector = FileSelector("DMA")
    dma_selector.select_files()
    dma_path = dma_selector.visualize_and_select()

    # --- 追加: 計算条件フォーム ---
    import tkinter as tk
    from tkinter import simpledialog

    class ParamForm(simpledialog.Dialog):
        def body(self, master):
            tk.Label(master, text="初期温度 (°C)").grid(row=0)
            tk.Label(master, text="最終温度 (°C)").grid(row=1)
            tk.Label(master, text="ポアソン比").grid(row=2)
            self.e1 = tk.Entry(master)
            self.e2 = tk.Entry(master)
            self.e3 = tk.Entry(master)
            self.e1.insert(0, "40")
            self.e2.insert(0, "200")
            self.e3.insert(0, "0.4")
            self.e1.grid(row=0, column=1)
            self.e2.grid(row=1, column=1)
            self.e3.grid(row=2, column=1)
            return self.e1

        def apply(self):
            self.result = {
                "init_temp": float(self.e1.get()),
                "final_temp": float(self.e2.get()),
                "poisson_ratio": float(self.e3.get()),
            }

    root = tk.Tk()
    root.withdraw()
    param_dialog = ParamForm(root, title="計算条件の入力")
    params = (
        param_dialog.result
        if param_dialog.result
        else {"init_temp": 40, "final_temp": 200, "poisson_ratio": 0.4}
    )
    root.destroy()

    # --- 計算実行 ---
    if ep_tma_path and peek_tma_path and dma_path:
        print("\nStep 4: Calculate residual stress")
        results = calculate_residual_stress(
            ep_tma_path=ep_tma_path,
            peek_tma_path=peek_tma_path,
            dma_path=dma_path,
            init_temp=params["init_temp"],
            final_temp=params["final_temp"],
            poisson_ratio=params["poisson_ratio"],
            save_figures=True,  # グラフを保存する場合はTrue
        )
    else:
        print("File selection was cancelled.")
