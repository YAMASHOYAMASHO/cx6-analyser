"""
このモジュールは、X線回折の3次元幾何学を可視化するためのスクリプトです。

主な機能:
    - 指定した角度パラメータ（α, ψ, 2θ, Φ）から回折に関わる各種ベクトル（入射光線、回折光線、結晶面法線など）を計算します。
    - 3Dプロットで回折幾何学の構造を描画し、各ベクトルや面、角度を可視化します。
    - 幾何学的検証（面上判定や角度計算）も含みます。

主な関数:
    calculate_vectors(alpha_deg, psi_deg, theta2_deg, Phi_deg):
        角度パラメータから各種ベクトルを計算します。
    plot_3d_diffraction(alpha_deg, psi_deg, theta2_deg, Phi_deg):
        3D回折幾何学の構造を描画します。

使用方法:
    スクリプトを実行すると、指定したパラメータに基づく回折幾何学の3D可視化が表示されます。

| 項目 | 記号 | 役割と説明 | 座標系における定義 (単位ベクトル) |
| :--- | :--- | :--- | :--- |
| **原点** | $\mathbf{O}$ | 座標系の中心、試料表面の点。 | $(0, 0, 0)$ |
| **$\mathbf{z}$軸ベクトル** | $\vec{P}$ | 半球の頂点（極軸）。慣習的に$z$軸正方向。 | $(0, 0, 1)$ |
| **入射光線** | $\vec{I}$ | $x$ 軸負方向から原点 $\mathbf{O}$ に向かう光線ベクトル。 | $(-1, 0, 0)$ |
| **回折光線** | $\vec{D}$ | 原点 $\mathbf{O}$ から出発する、回折後の光線ベクトル。 | $(D_x, D_y, D_z)$ (計算で決定) |
| **試料面法線** | $\vec{P'}$ | 試料表面に垂直なベクトル。$z$ 軸とのなす角が $\alpha$。 | $(\sin\alpha, 0, \cos\alpha)$ |
| **結晶面法線** | $\vec{Q}$ | 回折が起こる格子面に垂直なベクトル。 | $(Q_x, Q_y, Q_z)$ |
| **補助ベクトル** | $\vec{R}$ | 面 A ($yz$平面) と面 B の交線上のベクトル。 | $(0, \sin\Phi, \cos\Phi)$ |
| **角 $\alpha$** | $\angle POP'$ | $\vec{P}$ と $\vec{P'}$ のなす角。試料の傾斜角。 | - |
| **角 $\psi$** | $\angle P'OQ$ | $\vec{P'}$ と $\vec{Q}$ のなす角。 | - |
| **角 $\Phi$** | $\angle POR$ | $\vec{P}$ と $\vec{R}$ のなす角。 $\Phi = 90^\circ - \phi$ に相当。 | - |
| **角 $2\theta$** | $\angle IOD'$ | $\vec{I}$ と $\vec{D}$ のなす角 ($\vec{D}'$ は $\vec{D}$ の逆ベクトル）。回折角。 | $\vec{D} \cdot \vec{I} = -\cos(2\theta)$ |
| **面 A** | - | 入射光線（$x$軸）に垂直な面。 | $yz$ 平面 ($x=0$) |
| **面 B** | - | 入射光 $\vec{I}$、回折光 $\vec{D}$、および $\vec{Q}$ を含む面。 | - |
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider


# --- パラメータ設定 ---
alpha_deg = 10.0  # 試料傾斜角
psi_deg = 30.0  # P'とQのなす角
theta2_deg = 30.0  # 回折角
Phi_deg = 45.0  # PとRのなす角


def calculate_vectors(alpha_deg, psi_deg, theta2_deg, Phi_deg):
    """角度パラメータからベクトルを計算"""
    alpha = np.radians(alpha_deg)
    psi = np.radians(psi_deg)
    theta2 = np.radians(theta2_deg)
    Phi = np.radians(Phi_deg)

    # 基本ベクトル
    P = np.array([0, 0, 1])  # z軸ベクトル
    P_prime = np.array([np.sin(alpha), 0, np.cos(alpha)])  # 試料面法線
    I = np.array([1, 0, 0])  # 入射光線（(1,0,0)から(0,0,0)に向かう）
    R = np.array([0, np.sin(Phi), np.cos(Phi)])  # 補助ベクトル

    # 回折光線 D の計算
    # D・I = cos(2θ) を満たし、面B上にある
    # 面Bの法線: n_B = I × R = (0, -cos(Phi), sin(Phi))
    # 面B上の条件: D · n_B = 0 より -D_y*cos(Phi) + D_z*sin(Phi) = 0
    # よって D_y = D_z * tan(Phi)

    # Iベクトルは(1,0,0)から原点に向かうので、方向は(-1,0,0)
    # D・(-I) = cos(2θ) より D_x = cos(2θ)
    Dx = -np.cos(theta2)

    # |D| = 1 の条件: Dx^2 + Dy^2 + Dz^2 = 1
    # D_y = D_z * tan(Phi) を代入すると
    # Dx^2 + Dz^2 * (1 + tan^2(Phi)) = 1
    # Dx^2 + Dz^2 / cos^2(Phi) = 1

    if abs(Phi_deg - 90) < 0.1:  # Φ ≈ 90°の場合
        Dz = 0
        Dy = np.sqrt(max(0, 1 - Dx**2))
    else:
        Dz_sq = (1 - Dx**2) * np.cos(Phi) ** 2
        Dz = np.sqrt(max(0, Dz_sq))
        Dy = Dz * np.tan(Phi)

    D = np.array([Dx, Dy, Dz])
    D = D / np.linalg.norm(D)  # 正規化

    # 結晶面法線 Q の計算
    # Bragg条件: Qは入射ベクトル(-I)と回折ベクトルDの角二等分線方向
    # Q ∝ D - I (ここでIは方向ベクトルなので、-Iとの二等分)
    # より正確には Q ∝ D + (-I) = D - I
    I_direction = -I  # 原点に向かう方向
    Q_unnorm = D - I_direction  # (Dx-(-1), Dy, Dz) = (Dx+1, Dy, Dz)
    Q = Q_unnorm / np.linalg.norm(Q_unnorm)

    # Qが面B上にあることを確認
    n_B = np.cross(I, R)  # 面Bの法線
    # print(f"Q·n_B = {np.dot(Q, n_B):.6f} (should be ~0)")

    # ψの計算（確認用）
    # cos_psi_calc = np.dot(P_prime, Q)
    # psi_calc = np.degrees(np.arccos(np.clip(cos_psi_calc, -1, 1)))

    return P, P_prime, I, D, Q, R


def plot_3d_diffraction(alpha_deg, psi_deg, theta2_deg, Phi_deg):
    """3D回折幾何学のプロット"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # ベクトル計算
    P, P_prime, I, D, Q, R = calculate_vectors(alpha_deg, psi_deg, theta2_deg, Phi_deg)

    # --- 半球の描画 ---
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi / 2, 25)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(
        x_sphere, y_sphere, z_sphere, color="lightblue", alpha=0.1, linewidth=0
    )

    # 半球の枠線（点線）
    for i in range(0, len(u), 5):  # 経線
        ax.plot(
            x_sphere[i, :],
            y_sphere[i, :],
            z_sphere[i, :],
            "b:",
            linewidth=0.5,
            alpha=0.5,
        )
    for j in range(0, len(v), 3):  # 緯線
        ax.plot(
            x_sphere[:, j],
            y_sphere[:, j],
            z_sphere[:, j],
            "b:",
            linewidth=0.5,
            alpha=0.5,
        )

    # --- 面 A (yz平面) の描画 ---
    yy, zz = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(0, 1, 5))
    xx = np.zeros_like(yy)
    ax.plot_surface(xx, yy, zz, color="green", alpha=0.1, label="面A")

    # 面Aと半球の交線（半円）- x=0平面上の半円
    theta_A = np.linspace(-np.pi / 2, np.pi / 2, 100)
    x_A_circle = np.zeros_like(theta_A)
    y_A_circle = np.sin(theta_A)
    z_A_circle = np.cos(theta_A)
    ax.plot(x_A_circle, y_A_circle, z_A_circle, "g-", linewidth=2, label="面A交線")

    # --- 面 B の描画 ---
    # 面BはI, D, Qを含む面
    Phi = np.radians(Phi_deg)
    if abs(R[1]) > 0.001:  # sin(Phi) != 0
        y_b, z_b = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(0, 1, 5))
        x_b = y_b * 0  # 簡略化のため
        y_b_plane = (R[1] / R[2]) * z_b if abs(R[2]) > 0.001 else y_b
        ax.plot_surface(x_b, y_b_plane, z_b, color="cyan", alpha=0.1)

        # 面Bと半球の交線（半円）
        # 面Bの方程式: y*cos(Phi) - z*sin(Phi) = 0, または y = z*tan(Phi)
        theta_B = np.linspace(0, np.pi, 100)
        # 球面パラメータ: x=sin(t)cos(p), y=sin(t)sin(p), z=cos(t)
        # 面B上の条件: y/z = tan(Phi) より sin(p)/cos(p) = tan(Phi)
        # よって p = Phi
        z_B_circle = np.sin(theta_B) * np.cos(Phi)
        y_B_circle = np.sin(theta_B) * np.sin(Phi)
        x_B_circle = np.cos(theta_B)

    # --- 面 C (xz平面) の描画 ---
    xx, zz = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(0, 1, 5))
    yy = np.zeros_like(xx)

    # 面Cと半球の交線（半円）- y=0平面上の半円
    theta_A = np.linspace(-np.pi / 2, np.pi / 2, 100)
    y_A_circle = np.zeros_like(theta_A)
    x_A_circle = np.sin(theta_A)
    z_A_circle = np.cos(theta_A)
    ax.plot(x_A_circle, y_A_circle, z_A_circle, "r-", linewidth=2, label="面C交線")

    # --- 試料面の描画 ---
    L = 0.6
    alpha = np.radians(alpha_deg)
    # 試料の4隅（xy平面上に配置）
    corners = np.array(
        [[-L / 2, -L / 2, 0], [L / 2, -L / 2, 0], [L / 2, L / 2, 0], [-L / 2, L / 2, 0]]
    )
    # y軸周りにα回転
    R_alpha = np.array(
        [
            [np.cos(alpha), 0, np.sin(alpha)],
            [0, 1, 0],
            [-np.sin(alpha), 0, np.cos(alpha)],
        ]
    )
    corners_rot = corners @ R_alpha.T

    # 試料面の描画
    verts = [list(zip(corners_rot[:, 0], corners_rot[:, 1], corners_rot[:, 2]))]
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    poly = Poly3DCollection(verts, alpha=0.5, facecolor="orange", edgecolor="black")
    ax.add_collection3d(poly)

    # --- ベクトルの描画 ---
    origin = np.array([0, 0, 0])

    def draw_vector(v, color, label, scale=1.0, linestyle="-", linewidth=2):
        ax.quiver(
            0,
            0,
            0,
            v[0] * scale,
            v[1] * scale,
            v[2] * scale,
            color=color,
            arrow_length_ratio=0.15,
            linestyle=linestyle,
            linewidth=linewidth,
        )
        ax.text(
            v[0] * scale * 1.1,
            v[1] * scale * 1.1,
            v[2] * scale * 1.1,
            label,
            color=color,
            fontsize=11,
            fontweight="bold",
        )

    draw_vector(P, "black", r"$\vec{P}$", scale=1.0, linestyle="--", linewidth=1.5)
    draw_vector(P_prime, "red", r"$\vec{P'}$", scale=1.0)
    ax.plot(x_B_circle, y_B_circle, z_B_circle, "c-", linewidth=2, label="面B交線")
    draw_vector(Q, "blue", r"$\vec{Q}$", scale=1.0)
    draw_vector(R, "green", r"$\vec{R}$", scale=1.0, linestyle=":")

    # 入射光線(原点に向かう - (1,0,0)から(0,0,0)へ)
    ax.quiver(1, 0, 0, -1, 0, 0, color="purple", arrow_length_ratio=0.15, linewidth=2)
    ax.text(1.1, 0, 0, r"$\vec{I}$", color="purple", fontsize=11, fontweight="bold")

    # 回折光線(原点から出る)
    draw_vector(D, "orange", r"$\vec{D}$", scale=1.0)

    # --- 角度の表示 ---
    # α角の円弧
    theta_arc = np.linspace(0, np.radians(alpha_deg), 20)
    arc_r = 0.3
    ax.plot(
        arc_r * np.sin(theta_arc),
        np.zeros_like(theta_arc),
        arc_r * np.cos(theta_arc),
        "r--",
        linewidth=1.5,
    )
    ax.text(0.15, 0, 0.25, r"$\alpha$", color="red", fontsize=10)

    # Φ角の円弧
    phi_arc = np.linspace(0, np.radians(Phi_deg), 20)
    ax.plot(
        np.zeros_like(phi_arc),
        arc_r * np.sin(phi_arc),
        arc_r * np.cos(phi_arc),
        "g--",
        linewidth=1.5,
    )
    ax.text(0, 0.15, 0.25, r"$\Phi$", color="green", fontsize=10)

    # ψ角の円弧（P'とQのなす角）
    # P'とQを結ぶ平面上で円弧を描く
    # P'からQへの回転軸を求める
    if np.linalg.norm(np.cross(P_prime, Q)) > 1e-6:  # P'とQが平行でない場合
        # P'とQのなす角
        cos_psi_angle = np.dot(P_prime, Q) / (
            np.linalg.norm(P_prime) * np.linalg.norm(Q)
        )
        psi_angle = np.arccos(np.clip(cos_psi_angle, -1, 1))

        # 回転軸（P'とQの外積）
        rot_axis = np.cross(P_prime, Q)
        rot_axis = rot_axis / np.linalg.norm(rot_axis)

        # Rodriguesの回転公式でP'をQに向けて回転させる
        psi_arc_angles = np.linspace(0, psi_angle, 20)
        psi_arc_points = []
        for t in psi_arc_angles:
            # P'を回転軸周りにt回転
            K = np.array(
                [
                    [0, -rot_axis[2], rot_axis[1]],
                    [rot_axis[2], 0, -rot_axis[0]],
                    [-rot_axis[1], rot_axis[0], 0],
                ]
            )
            R_t = np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)
            rotated = R_t @ P_prime * arc_r
            psi_arc_points.append(rotated)

        psi_arc_points = np.array(psi_arc_points)
        ax.plot(
            psi_arc_points[:, 0],
            psi_arc_points[:, 1],
            psi_arc_points[:, 2],
            "b--",
            linewidth=1.5,
        )

        # ψのラベル位置（円弧の中点付近）
        mid_point = psi_arc_points[len(psi_arc_points) // 2]
        ax.text(
            mid_point[0] * 1.3,
            mid_point[1] * 1.3,
            mid_point[2] * 1.3,
            r"$\psi$",
            color="blue",
            fontsize=10,
        )

    # --- 軸設定 ---
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([0, 1.2])
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Z", fontsize=12)

    # 軸のアスペクト比を等しく設定（まん丸の球にする）
    ax.set_box_aspect([2, 2, 1])

    # 角度の計算と検証
    cos_psi = np.dot(P_prime, Q)
    psi_calc = np.degrees(np.arccos(np.clip(cos_psi, -1, 1)))

    # 2θの計算（IとDのなす角）
    # Iは(1,0,0)から原点に向かうので方向は(-1,0,0)
    I_direction = -I
    cos_2theta = np.dot(D, I_direction)
    theta2_calc = np.degrees(np.arccos(np.clip(cos_2theta, -1, 1)))

    # 面Bの法線とQの内積（これが0ならQは面B上）
    n_B = np.cross(I, R)
    Q_on_planeB = np.dot(Q, n_B)

    title = f"X線回折幾何学\n"
    title += f"α={alpha_deg:.1f}°, ψ={psi_calc:.1f}°, Φ={Phi_deg:.1f}°, 2θ={theta2_calc:.1f}°\n"
    title += f"Q·n_B={Q_on_planeB:.4f} (面B上の確認)"

    ax.view_init(elev=20, azim=-60)
    ax.grid(True, alpha=0.3)

    return fig, ax


# 初期プロット
fig, ax = plot_3d_diffraction(alpha_deg, psi_deg, theta2_deg, Phi_deg)

plt.show()
