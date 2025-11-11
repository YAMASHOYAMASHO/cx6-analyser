"""
TMA/DMAデータのプロット用ユーティリティ関数

residual_stress_calculater.pyから抽出したプロット機能を
独立した関数として提供します。
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from typing import Optional, Sequence, Tuple


def create_tma_dma_overlay_figure(
    ep_temp: np.ndarray,
    ep_expansion: np.ndarray,
    peek_temp: np.ndarray,
    peek_expansion: np.ndarray,
    dma_temp: np.ndarray,
    dma_ereal: np.ndarray,
    title: str = "TMA and DMA Data",
    figsize: Tuple[float, float] = (10, 6),
):
    """
    TMA（EP/PEEK）とDMAを重ね合わせた図を作成する関数

    左Y軸: DMA貯蔵弾性率（対数スケール）
    右Y軸: TMA熱膨張率（線形スケール）

    Parameters:
    -----------
    ep_temp : np.ndarray
        EP（フィルム）の温度データ
    ep_expansion : np.ndarray
        EP（フィルム）の熱膨張率データ (%)
    peek_temp : np.ndarray
        PEEK（基板）の温度データ
    peek_expansion : np.ndarray
        PEEK（基板）の熱膨張率データ (%)
    dma_temp : np.ndarray
        DMAの温度データ
    dma_ereal : np.ndarray
        DMAの貯蔵弾性率データ (MPa)
    title : str, optional
        グラフのタイトル
    figsize : Tuple[float, float], optional
        図のサイズ (width, height)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        作成された図
    ax_dma : matplotlib.axes.Axes
        DMA用の軸（左Y軸、対数スケール）
    ax_tma : matplotlib.axes.Axes
        TMA用の軸（右Y軸、線形スケール）

    Example:
    --------
    >>> fig, ax_dma, ax_tma = create_tma_dma_overlay_figure(
    ...     ep_temp, ep_expansion,
    ...     peek_temp, peek_expansion,
    ...     dma_temp, dma_ereal
    ... )
    >>> plt.show()
    """
    fig, ax_dma = plt.subplots(figsize=figsize)
    ax_dma.set_yscale("log", base=10)
    ax_tma = ax_dma.twinx()

    # TMAデータをプロット（右Y軸）
    ax_tma.plot(ep_temp, ep_expansion, label="EP TMA (Film)", color="blue", linewidth=2)
    ax_tma.plot(
        peek_temp,
        peek_expansion,
        label="PEEK TMA (Substrate)",
        color="orange",
        linewidth=2,
    )
    ax_tma.set_xlabel("Temperature (°C)", fontsize=14)
    ax_tma.set_ylabel("Thermal expansion (%)", fontsize=14)

    # DMAデータをプロット（左Y軸）
    ax_dma.plot(dma_temp, dma_ereal, label="DMA E'", color="green", linewidth=2)
    ax_dma.set_ylabel("E' (MPa)", fontsize=14)

    # 凡例を統合
    lines_dma, labels_dma = ax_dma.get_legend_handles_labels()
    lines_tma, labels_tma = ax_tma.get_legend_handles_labels()
    ax_dma.legend(
        lines_dma + lines_tma, labels_dma + labels_tma, loc="best", fontsize=11
    )

    ax_dma.grid(True, alpha=0.3, which="both")
    plt.title(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    return fig, ax_dma, ax_tma


def plot_tma_comparison(
    ax: Axes,
    tma_data_list: Sequence[Tuple[np.ndarray, np.ndarray]],
    labels: Sequence[str],
    selected_index: Optional[int] = None,
    title: str = "TMA Data Comparison",
):
    """
    複数のTMAデータを比較表示する関数

    選択されたデータは赤色・太線で強調表示されます。

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        プロット対象のAxes
    tma_data_list : Sequence[Tuple[np.ndarray, np.ndarray]]
        TMAデータのリスト。各要素は (温度, 熱膨張率) のタプル
    labels : Sequence[str]
        各データのラベル
    selected_index : Optional[int], optional
        強調表示するデータのインデックス（0始まり）
    title : str, optional
        グラフのタイトル

    Example:
    --------
    >>> fig, ax = plt.subplots()
    >>> plot_tma_comparison(
    ...     ax,
    ...     [(temp1, exp1), (temp2, exp2), (temp3, exp3)],
    ...     ["Sample 1", "Sample 2", "Sample 3"],
    ...     selected_index=1
    ... )
    >>> plt.show()
    """
    for i, ((temp, expansion), label) in enumerate(zip(tma_data_list, labels)):
        is_selected = selected_index is not None and i == selected_index
        color = "red" if is_selected else f"C{i}"
        linewidth = 3 if is_selected else 1.5
        alpha = 1.0 if is_selected else 0.6
        zorder = 10 if is_selected else 1

        ax.plot(
            temp,
            expansion,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            zorder=zorder,
        )

    ax.set_xlabel("Temperature (°C)", fontsize=12)
    ax.set_ylabel("Thermal expansion (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")


def plot_dma_comparison(
    ax: Axes,
    dma_data_list: Sequence[Tuple[np.ndarray, np.ndarray]],
    labels: Sequence[str],
    selected_index: Optional[int] = None,
    title: str = "DMA Data Comparison",
):
    """
    複数のDMAデータを比較表示する関数（対数スケール）

    選択されたデータは赤色・太線で強調表示されます。

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        プロット対象のAxes
    dma_data_list : Sequence[Tuple[np.ndarray, np.ndarray]]
        DMAデータのリスト。各要素は (温度, 貯蔵弾性率) のタプル
    labels : Sequence[str]
        各データのラベル
    selected_index : Optional[int], optional
        強調表示するデータのインデックス（0始まり）
    title : str, optional
        グラフのタイトル

    Example:
    --------
    >>> fig, ax = plt.subplots()
    >>> plot_dma_comparison(
    ...     ax,
    ...     [(temp1, ereal1), (temp2, ereal2)],
    ...     ["Sample 1", "Sample 2"],
    ...     selected_index=0
    ... )
    >>> plt.show()
    """
    ax.set_yscale("log")

    for i, ((temp, ereal), label) in enumerate(zip(dma_data_list, labels)):
        is_selected = selected_index is not None and i == selected_index
        color = "red" if is_selected else f"C{i}"
        linewidth = 3 if is_selected else 1.5
        alpha = 1.0 if is_selected else 0.6
        zorder = 10 if is_selected else 1

        ax.plot(
            temp,
            ereal,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            zorder=zorder,
        )

    ax.set_xlabel("Temperature (°C)", fontsize=12)
    ax.set_ylabel("E' (MPa)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="best")


def plot_thermal_expansion_coefficients(
    temp_range: np.ndarray,
    peek_alpha: np.ndarray,
    ep_alpha: np.ndarray,
    title: str = "Thermal Expansion Coefficients",
    figsize: Tuple[float, float] = (10, 6),
):
    """
    線熱膨張係数をプロットする関数

    Parameters:
    -----------
    temp_range : np.ndarray
        温度範囲
    peek_alpha : np.ndarray
        PEEK（基板）の線熱膨張係数 (1/°C)
    ep_alpha : np.ndarray
        EP（フィルム）の線熱膨張係数 (1/°C)
    title : str, optional
        グラフのタイトル
    figsize : Tuple[float, float], optional
        図のサイズ

    Returns:
    --------
    fig : matplotlib.figure.Figure
        作成された図
    ax : matplotlib.axes.Axes
        プロット用の軸

    Example:
    --------
    >>> fig, ax = plot_thermal_expansion_coefficients(
    ...     temp_range, peek_alpha, ep_alpha
    ... )
    >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        temp_range,
        peek_alpha * 1e6,
        label="PEEK α (Substrate)",
        color="blue",
        linewidth=2,
    )
    ax.plot(
        temp_range, ep_alpha * 1e6, label="EP α (Film)", color="orange", linewidth=2
    )

    ax.set_xlabel("Temperature (°C)", fontsize=14)
    ax.set_ylabel("Thermal expansion coefficient (×10⁻⁶ /°C)", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.title(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    return fig, ax


def plot_residual_stress(
    temp_points: np.ndarray,
    stress_vs_temp: np.ndarray,
    title: str = "Residual Stress vs Temperature",
    figsize: Tuple[float, float] = (10, 6),
):
    """
    残留応力の温度依存性をプロットする関数

    Parameters:
    -----------
    temp_points : np.ndarray
        温度点
    stress_vs_temp : np.ndarray
        各温度での残留応力 (MPa)
    title : str, optional
        グラフのタイトル
    figsize : Tuple[float, float], optional
        図のサイズ

    Returns:
    --------
    fig : matplotlib.figure.Figure
        作成された図
    ax : matplotlib.axes.Axes
        プロット用の軸

    Example:
    --------
    >>> fig, ax = plot_residual_stress(temp_points, stress_vs_temp)
    >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(temp_points, stress_vs_temp, color="red", s=20, alpha=0.6)
    ax.plot(temp_points, stress_vs_temp, color="red", linewidth=2, alpha=0.8)

    ax.set_xlabel("Temperature (°C)", fontsize=14)
    ax.set_ylabel("Residual Stress (MPa)", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.title(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    return fig, ax
