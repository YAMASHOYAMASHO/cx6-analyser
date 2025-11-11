"""
このモジュールは、X線回折実験における浸透深さを計算するためのものです。

関数:
    depth(alpha, phi, twotheta, EP=False, EP_tickness=200):
        指定されたパラメータに基づいて浸透深さを計算します。

定数:
    - プランク定数、光速、アボガドロ数などの物理定数。
    - 水素、炭素、窒素、酸素の物質固有の定数。
    - PEEKおよびEPの事前定義された物性値。

使用方法:
    このスクリプトは、スタンドアロンプログラムとして実行可能で、
    各種角度や材料に対する浸透深さを計算および可視化します。
"""

lam = int(input("波長 (A): ")) * 1e-10  # m 波長
# 波長を入力してもらう

# 定数
C_H = 0.0127
D_H = 0.466e-5
Z_H = 1
M_H = 1.00794

C_C = 1.22
D_C = 0.0142
Z_C = 6
M_C = 12.011

C_N = 2.05
D_N = 0.0317
Z_N = 7
M_N = 14.00674

C_O = 3.18
D_O = 0.0654
Z_O = 8
M_O = 15.9994

import math

# コンプトン散乱の計算
sigma_T = 6.7e-25  # cm^2
h = 6.626e-34  # J*s プランク定数
c = 3e8  # m/s 光速
E = h * c / lam  # J エネルギー
E_eV = E / 1.6e-19  # keV エネルギー
E_keV = E_eV / 1e3  # keV エネルギー
alpha = E_keV / 511

term1 = (1 + alpha) / alpha**2
term2 = (2 * (1 + alpha) / (1 + 2 * alpha)) - (math.log(1 + 2 * alpha) / alpha)
term3 = math.log(1 + 2 * alpha) / (2 * alpha)
term4 = (1 + 3 * alpha) / (1 + 2 * alpha) ** 2

sigma_c = sigma_T * (3 / 4) * (term1 * term2 + term3 - term4)

# 各物質の定数
N_A = 6.022e23  # mol^-1 アボガドロ数
N_rho_H = N_A * Z_H / M_H  # 電子密度 (mol/cm^3)
N_rho_C = N_A * Z_C / M_C  # 電子密度 (mol/cm^3)
N_rho_N = N_A * Z_N / M_N  # 電子密度 (mol/cm^3)
N_rho_O = N_A * Z_O / M_O  # 電子密度 (mol/cm^3)

# 質量吸収係数 Victoreen's equation
# mu_rho = C * lam^3 - D * lam^4 + N_rho * sigma_c
lam_A = lam * 1e10  # Angstrom
mu_rho_H = C_H * lam_A**3 - D_H * lam_A**4 + N_rho_H * sigma_c
mu_rho_C = C_C * lam_A**3 - D_C * lam_A**4 + N_rho_C * sigma_c
mu_rho_N = C_N * lam_A**3 - D_N * lam_A**4 + N_rho_N * sigma_c
mu_rho_O = C_O * lam_A**3 - D_O * lam_A**4 + N_rho_O * sigma_c

# results
print(f"波長: {lam} m")
print(f"エネルギー: {E_keV:.2f} keV")
print(f"μ(H): {mu_rho_H:.2f} cm^2/g")
print(f"μ(C): {mu_rho_C:.2f} cm^2/g")
print(f"μ(N): {mu_rho_N:.2f} cm^2/g")
print(f"μ(O): {mu_rho_O:.2f} cm^2/g")

# PEEK C19H12O3
PEEK_total = 19 * M_C + 12 * M_H + 3 * M_O
PEEK_C = 19 * M_C / PEEK_total
PEEK_H = 12 * M_H / PEEK_total
PEEK_O = 3 * M_O / PEEK_total

PEEK_rho = 1.3  # g/cm^3

PEEK_mu_rho = PEEK_C * mu_rho_C + PEEK_H * mu_rho_H + PEEK_O * mu_rho_O
PEEK_mu = PEEK_mu_rho * PEEK_rho  # cm^-1

print(f"PEEK μ: {PEEK_mu:.2f} cm^-1")

# jER (C18H20O3)nC21H23O4
jER_cycle = M_C * 18 + M_H * 20 + M_O * 3
jER_base = M_C * 21 + M_H * 23 + M_O * 4

jER_weight = 370  # g/mol

n = (jER_weight - jER_base) / jER_cycle  # 繰り返し単位数
print(f"jER n: {n:.2f}")

# DDM C13H14N2

# EP = 2 * jER + DDM
EP_total = 2 * jER_weight + 13 * M_C + 14 * M_H + 2 * M_N
EP_C = (2 * (18 * n + 21) + 13) * M_C / EP_total
EP_H = (2 * (20 * n + 23) + 14) * M_H / EP_total
EP_N = 2 * M_N / EP_total
EP_O = 2 * (3 * n + 4) * M_O / EP_total

EP_rho = 1.19  # g/cm^3

EP_mu_rho = EP_C * mu_rho_C + EP_H * mu_rho_H + EP_N * mu_rho_N + EP_O * mu_rho_O
EP_mu = EP_mu_rho * EP_rho  # cm^-1

print(f"EP μ: {EP_mu:.2f} cm^-1")


def depth(alpha, phi, twotheta, EP=False, EP_tickness=200):
    """
    alpha: 入射角 (deg)
    phi: 方位角 (deg)
    twotheta: 2theta (deg)
    EP: EPの時はTrueを指定
    EP_tickness: EPの厚さ (um)
    """

    EP_tickness = EP_tickness * 1e-4  # cm

    alpha = math.radians(alpha)
    phi = math.radians(phi)
    twotheta = math.radians(twotheta)

    term1 = 1 / math.sin(alpha)
    term2 = math.sin(twotheta) * math.cos(alpha) * math.sin(phi)
    term3 = math.cos(twotheta) * math.sin(alpha)
    term4 = 1 / (term2 - term3)
    term5 = term1 + term4

    if EP:
        # EPの時はPEEKとEPの質量吸収係数を考慮する
        T = 1 / EP_mu / term5
        if T > EP_tickness:
            T = (1 - 0.02 * (EP_mu - PEEK_mu) * term5) / PEEK_mu / term5

    else:
        # PEEKの時はPEEKの質量吸収係数を考慮する
        T = 1 / PEEK_mu / term5

    return T

if __name__ == "__main__":
    
    from matplotlib import pyplot as plt
    import numpy as np
    import matplotlib.cm as cm

    for i, alpha in enumerate([0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5]):
        for phi in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
            T = depth(alpha, phi, 12.2)
            if phi == 10:
                plt.scatter(phi, T, c=cm.tab10(i), label=f"alpha={alpha}")
            else:
                plt.scatter(phi, T, c=cm.tab10(i))

    plt.xlabel("phi (deg)")
    plt.ylabel("T (cm)")

    plt.legend()