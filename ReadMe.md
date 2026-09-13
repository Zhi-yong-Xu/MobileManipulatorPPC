# 移动机械臂 PPC 控制器族

面向 SummitXL + Panda 移动机械臂的预设性能控制（PPC）实现，包含六个消融版本，完整记录从基线到修复版的演进路径。

## 目录

- [概览](#概览)
- [版本矩阵](#版本矩阵)
- [运动学与动力学模型](#运动学与动力学模型)
- [轨迹生成](#轨迹生成)
- [零空间参考信号](#零空间参考信号)
- [预设性能控制](#预设性能控制)
- [力矩接口分配](#力矩接口分配)
- [数值实现](#数值实现)
- [诊断量](#诊断量)
- [符号—代码映射](#符号代码映射)
- [附录：DSC 对照基线](#附录dsc-对照基线)

---

## 概览

六个文件，一个控制器族：

| 简称 | 文件 | 角色 | 核心思想 |
|---|---|---|---|
| v0 | `ppc01_baseline.py` | 基线 | 变换双重饱和 + 输出限幅 + 无限位回避 + 可切换 DSC |
| v1 | `ppc02_mu_unclip.py` | μ 去饱和 | 去除 μ 外层 clip [A] + 首次引入限位排斥/刹车 [B] |
| v2 | `ppc03_reinit.py` | 纯变换 + 重 init | 纯微分同胚 + 域外重初始化 [A+]（固定开）+ P1/P2 前提检查 |
| v3 | `ppc04_noclip.py` | 去输出限幅 | 去除输出限幅 [X2] + 漏斗收紧 + REINIT/裸 PPC 开关 [X3] |
| v4 | `ppc05_fixed.py` | 极点修复 | 极点修复 [FIX1] + 屏障强度 [FIXB] + M 加权 settle [FIXA] + 审计 |
| 1L | `ppc_onelayer_abl.py` | 一层消融 | 仅保留速度层；限幅与宽漏斗刻意还原；唯一变量 = 层数 |

### 默认开关设置

| 文件 | 输出限幅 | 域外处理 | DAMP_IN_M | γ₁ / γ₂ | ρ₁∞ 臂 | ρ₂∞ 臂 | Settle |
|---|---|---|---|---|---|---|---|
| `ppc01_baseline.py` | ON | 静默 | — | 0.4 / 3.0 | 0.12 | 0.25 | 均匀 PD |
| `ppc02_mu_unclip.py` | ON | 静默 | — | 0.4 / 3.0 | 0.12 | 0.25 | 均匀 PD |
| `ppc03_reinit.py` | ON | REINIT 固定 | — | 0.4 / 3.0 | 0.12 | 0.25 | 均匀 PD |
| `ppc04_noclip.py` | OFF | REINIT 开关 | — | 0.4 / 3.0 | 0.025 | 0.10 | 均匀 PD |
| `ppc05_fixed.py` | OFF | REINIT 开关 | `True` | 0.25 / 8.0 | 0.04 | 0.10 | M 加权 |
| `ppc_onelayer_abl.py` | ON | REINIT（仅 L2） | — | — / 3.0 | 0.12（诊断） | 0.25 | 均匀 PD |

---

## 运动学与动力学模型

### 状态与输入

```latex
q = [x_b, y_b, psi, q_{a,1}, ..., q_{a,7}]^T in R^10
tau = [tau_{w,1}, ..., tau_{w,4}, tau_{a,1}, ..., tau_{a,7}]^T in R^11
zeta = [xdot_b, ydot_b, psidot]^T in R^3
```

关节限位 `q_{a,j} in [ell_j, h_j]`。`J^{+}` 为截断伪逆（`rcond=1e-6`），`1{.}` 为示性函数。

### 差速底盘

```latex
R_z(psi) = [cos(psi)  -sin(psi)  0;  sin(psi)  cos(psi)  0;  0  0  1]
R_{w2b} = R_z^T(psi)
u_body = R_{w2b} zeta
```

```latex
r = 0.12,  L = 0.2225 + 0.2045 = 0.427
K = [1  1   L;  1  -1  -L;  1  -1  L;  1  1  -L]
omega_w = J_w u_body,  J_w = (1/r) K
```

### 末端雅可比

```latex
xdot_e = [pdot_ee; omega_ee] = J_e(q) qdot,  J_e = [J_b | J_a] in R^{6 x 10}
```

```latex
J_b^(lin) = [e_x, e_y, z_hat x (p_ee - p_b)]
J_b^(ang) = [0_{2x3}, z_hat]^T
```

```latex
J_a^(lin)[.,i] = z_i x (p_ee - p_i)
J_a^(ang)[.,i] = z_i
J_p = J_e[1:3, :],  J_r = J_e[4:6, :]
```

### 姿态误差（世界系，最短路径）

```latex
q_err = Q* (x) Q_ee^{-1} = (eta, epsilon)
q_err <- -q_err  if eta < 0
```

```latex
e_R = theta * epsilon_hat
theta = 2 * atan2(||epsilon||, eta)
epsilon_hat = epsilon / ||epsilon||  (||epsilon|| = 0 => e_R = 0)
```

### 动力学

```latex
M(q) qddot + C(q, qdot) qdot + G(q) = B(q) tau + tau_passive(q, qdot) + d(t)
```

```latex
qddot = M^{-1} (B tau - h + d),  h := C qdot + G - tau_passive
```

`h` 逐 DOF 读自 `qfrc_bias - qfrc_passive`。集中扰动为

```latex
d = -J_e^T F_ext   (接触)
  + tau_fric       (摩擦)
  + (I - B B^+) B tau  (平面解耦残差)
  + tau_sat        (饱和)
  + O(dt)          (离散化)
```

### 执行器映射

```latex
B = [T_f R_{w2b}   0_{3x7};   0_{7x4}   I_7]
T_f = (r/4) K diag(1, 1, 1/L^2)
```

```latex
tau_phys = sat_[ctrlrange](tau_cmd)   (仅 ctrllimited 执行器；本模型 0/11)
```

力分配精确性——由 `K^T K = 4 diag(1, 1, L^2)`：

```latex
(1/r) K^T T_f = I_3
=> R_z(psi) (1/r) K^T tau_w = tau_sigma[0:3]
```

---

## 轨迹生成

### 参考轨迹（全版本一致）

```latex
p_d(t) = [cos(0.2 t); sin(0.2 t); 0.7]
pdot_d = 0.2 [-sin(0.2 t); cos(0.2 t); 0]
Q* = [0, 1, 0, 0]^T,  omega_d = 0
```

```latex
e_p = p_d - p_ee,  e_R 由姿态误差公式给出
```

### 分通道向量饱和（全版本一致）

```latex
sat_a(x) = (a / max(||x||, a)) x
v_c = sat_{0.6}(pdot_d + 10 e_p)
omega_c = sat_{1.0}(omega_d + 2 e_R)
```

### 阻尼最小二乘主任务（全版本一致）

```latex
qdot_task = argmin_{qdot}  || J_e qdot - [v_c; omega_c] ||^2 + lambda^2 ||qdot||^2
         => qdot_task = J_e^T (J_e J_e^T + lambda^2 I_6)^{-1} [v_c; omega_c]
```

```latex
w(q) = sqrt(det(J_p J_p^T))
lambda^2 = lambda_min^2 + lambda_0^2 (1 - min(w / w_thr, 1))^2
```

其中 `lambda_min = 1e-3`，`lambda_0 = 0.05`，`w_thr = 0.02`。

### Slew 限变化（全版本一致，`S = 2`）

```latex
qdot_des(t) = qdot_des(t - dt)
            + min(1, S dt / ||qdot_raw - qdot_des(t - dt)||_inf)
              * (qdot_raw - qdot_des(t - dt))
```

---

## 零空间参考信号

优先级：**任务 1**（末端跟踪，主任务）→ **任务 2**（奇异规避）→ **任务 3**（限位回避）。任务 2 与任务 3 通过任务 1 的零空间投影。

### 零空间投影算子（全版本一致）

```latex
N = I_10 - J_e^+ J_e,  J_e N = 0 (精确),  N^2 = N
```

### 任务 2 —— 奇异规避（全版本一致）

```latex
g_{w,k} = (w(q + eps_g e_k) - w(q)) / eps_g,  eps_g = 1e-5
qdot_null^(w) = k_eff N (g_w / ||g_w||)
```

```latex
k_eff = 0.5 clip((0.5 - ||e_p||) / 0.3, 0, 1)
启用 <=> k_eff > 1e-6  and  w > 0.3 w_thr  and  ||g_w|| > 1e-12
```

### 任务 3 —— 限位回避

**`ppc01_baseline.py`（v0）：** 该模块不存在。

```latex
qdot_null^(lim) = 0_7
```

**`ppc02_mu_unclip.py` / `ppc03_reinit.py` / `ppc04_noclip.py` / `ppc05_fixed.py` / `ppc_onelayer_abl.py`（v1+）：** 排斥 + 刹车。

```latex
s_j^- = (d_m - (q_{a,j} - ell_j)) / d_m
s_j^+ = (d_m - (h_j - q_{a,j})) / d_m
d_m = 0.15,  k_lim = 2
```

```latex
v_rep,j = k_lim (s_j^-)^2 1{q_{a,j} - ell_j < d_m}
        - k_lim (s_j^+)^2 1{h_j - q_{a,j} < d_m}
```

```latex
(P qdot)_{3+j} = qdot_{3+j} * 1{ not(h_j - q_{a,j} < d_b and qdot_{3+j} > 0)
                           and not(q_{a,j} - ell_j < d_b and qdot_{3+j} < 0) }
d_b = 0.05
```

### 合成

**`ppc01_baseline.py`（v0）：**

```latex
qdot_raw = qdot_task + qdot_null^(w)
```

**`ppc02_mu_unclip.py` / `ppc03_reinit.py` / `ppc04_noclip.py` / `ppc05_fixed.py` / `ppc_onelayer_abl.py`（v1+）：**

```latex
qdot_raw = P ( qdot_task + qdot_null^(w) + qdot_null^(lim) )
```

> **注。** 刹车算子作用于零空间投影之后，故仅当刹车激活时 `J_e P qdot_null != 0`。残差并入 (11) 的 `d`。

### 差异表

| 特性 | v0 | v1 | v2 | v3 | v4 | 1L |
|---|---|---|---|---|---|---|
| 任务 1（末端跟踪） | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 任务 2（奇异规避） | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 任务 3（限位回避） | **✗** | ✓ | ✓ | ✓ | ✓ | ✓ |
| 合成 | qdot_task + qdot_null^(w) | P(... + v_rep) | 同 v1 | 同 v1 | 同 v1 | 同 v1 |

---

## 预设性能控制

### 参考积分器与误差定义

**`ppc01_baseline.py` / `ppc02_mu_unclip.py` / `ppc03_reinit.py` / `ppc04_noclip.py` / `ppc05_fixed.py`（两层版）：**

```latex
q_d(0) = q(0) 精确 (P1)
qdot_d = qdot_des
e_1 = q - q_d,  e_{1,2} <- wrap(e_{1,2})
```

```latex
wrap(a) = (a + pi) mod (2 pi) - pi
```

**`ppc_onelayer_abl.py`（1L）：** 控制回路内无 `q_d`、`e_1`、`mu_1`。保留诊断积分器但从不反馈：

```latex
q_d^{diag} <- int(qdot_des)
e_1^{diag} = q - q_d^{diag}
```

精确关系：`e_1^{diag}(t) = int_0^t (qdot - qdot_des) d tau = int_0^t (e_2 + xi) d tau`，其中 `xi := alpha_f - alpha`。

### 漏斗函数

```latex
rho(t) = (rho_0 - rho_inf) exp(-alpha' t / (T - t)) + rho_inf
alpha' = 2,  T = 10 s
```

初值（全版本一致）：

```latex
rho_{1,0} = [0.20, 0.20, 0.30 | 0.25 x7]
rho_{2,0} = [2.5,  2.5,  2.0  | 3.0  x7]
```

终值（差异）：

**`ppc01_baseline.py` / `ppc02_mu_unclip.py` / `ppc03_reinit.py`：**

```latex
rho_{1,inf} = [0.15, 0.15, 0.20 | 0.12 x7]
rho_{2,inf} = [0.20, 0.20, 0.20 | 0.25 x7]
```

**`ppc04_noclip.py`：**

```latex
rho_{1,inf} = [0.05, 0.05, 0.05 | 0.025 x7]
rho_{2,inf} = [0.10, 0.10, 0.10 | 0.10  x7]
```

**`ppc05_fixed.py`：**

```latex
rho_{1,inf} = [0.05, 0.05, 0.05 | 0.04  x7]
rho_{2,inf} = [0.10, 0.10, 0.10 | 0.10  x7]
```

**`ppc_onelayer_abl.py`：**

```latex
rho_{2,inf} = [0.20, 0.20, 0.20 | 0.25 x7]
rho_{1,inf}^{diag} = [0.15, 0.15, 0.20 | 0.12 x7]   (仅作绘图参考线)
```

### 理想变换（v2–v4 与 1L 实现的数学对象）

```latex
phi = e / rho(t)
T(phi) = ln((0.9 + phi) / (0.9 - phi)) = 2 artanh(phi / 0.9)
```

```latex
D_t = {e : |e| < 0.9 rho(t)},  T : D_t -> R 严格微分同胚
```

```latex
T'(phi) = 1.8 / (0.81 - phi^2)  >=  k_T := T'(0) = 1.8 / 0.81 = 2.2222
lim_{phi -> +-0.9} T'(phi) = +inf
```

```latex
e = 0.9 rho tanh(mu / 2),  sign(e) = sign(mu)
```

### 各版本实际变换

**`ppc01_baseline.py`（v0，双重饱和）：**

```latex
phi_c = clip(e / rho, -0.899, 0.899)
mu = clip(T(phi_c), -1, 1)
```

```latex
d mu / d e = T'(phi) / rho,   |phi| <= 0.9 tanh(1/2) = 0.4159
          = 0,                0.4159 < |phi| < 0.9
```

**`ppc02_mu_unclip.py`（v1）：**

```latex
phi_c = clip(e / rho, -0.899, 0.899)
mu = T(phi_c)
|mu| <= T(0.899) = ln 1799 ~ 7.5
```

**`ppc03_reinit.py` / `ppc04_noclip.py` / `ppc05_fixed.py` / `ppc_onelayer_abl.py`（v2+）：**

```latex
mu = T(e / rho)   (零截断，(39)–(42) 全部成立)
```

### Level-1 虚拟控制

**`ppc01_baseline.py` / `ppc02_mu_unclip.py` / `ppc03_reinit.py` / `ppc04_noclip.py`：**

```latex
alpha = qdot_des - gamma_1 mu_1,  gamma_1 = 0.4
```

**`ppc05_fixed.py`：**

```latex
alpha = qdot_des - gamma_1 mu_1,  gamma_1 = 0.25
```

**`ppc_onelayer_abl.py`：**

```latex
INJECT = 0 :  alpha = qdot_des
INJECT = 1 :  alpha = qdot_des - K_I z,  K_I = 2,  zdot = e_2,  |z| <= 0.5
```

> **恒等式。** 当 `z(0) = e_1(0) = 0`，有 `z(t) = int_0^t e_2 d tau = e_1(t)`。注入版等价于 `alpha = qdot_des - K_I e_1`——即去除变换的两层律（`gamma_1 -> K_I`）。

### 命令滤波器（全版本一致）

```latex
alphadot_f = -lambda_f (alpha_f - alpha),  lambda_f = 20,  alpha_f(0) = 0
xidot = -lambda_f xi - alphadot
```

```latex
e_2 = qdot - alpha_f
```

### Level-2 力矩律

**旧律（`ppc01_baseline.py` / `ppc02_mu_unclip.py` / `ppc03_reinit.py` / `ppc04_noclip.py` / `ppc_onelayer_abl.py`）：**

```latex
s_2 = tanh(mu_2 / delta),  delta = 0.5
K_b = 2 + 0.02 min(||qdot||, 100)^2
gamma_2 = 3.0,  epsilon = 0.15
```

```latex
tau_sigma = h + M (alphadot_f - gamma_2 mu_2 - epsilon s_2) - K_b s_2
```

**新律（`ppc05_fixed.py`，DAMP_IN_M=True，默认）：**

```latex
k_r = 0.5 + 0.005 min(||qdot||, 100)^2
gamma_2 = 8.0,  epsilon = 0.15
```

```latex
tau_sigma = h + M (alphadot_f - gamma_2 mu_2 - epsilon s_2 - k_r s_2)
```

**A/B 对照律（`ppc05_fixed.py`，DAMP_IN_M=False）：**

```latex
K_b = 2 + 0.02 min(||qdot||, 100)^2,  gamma_2 = 8.0
tau_sigma = h + M (alphadot_f - gamma_2 mu_2 - epsilon s_2) - K_b s_2
```

### 闭环误差动力学

位置层（两层版公共）：

```latex
edot_1 = e_2 + xi - gamma_1 mu_1
mudot_1 = T'(phi_1) (edot_1 / rho_1 - phi_1 sigma_1)
```

速度层：

```latex
旧律: edot_{2,i} = -gamma_2 mu_{2,i}
                 - (epsilon + K_b / M_{ii}) tanh(mu_{2,i} / delta)
                 + d_i / M_{ii}
```

```latex
新律: edot_{2,i} = -gamma_2 mu_{2,i}
                 - (epsilon + k_r) tanh(mu_{2,i} / delta)
                 + d_i / M_{ii}
```

线性化极点（`mu_2 -> 0`，`T' -> k_T`，`d/dmu tanh(mu/delta)|_0 = 2/delta`）：

```latex
lambda_i^old = (gamma_2 + 2 epsilon) k_T / rho_{2,inf}    [经 M，通道一致]
             + 2 K_{b,0} k_T / (rho_{2,inf} M_{ii})      [裸 K_b，除 M_{ii}]
             = 73.3 + 88.9 / M_{ii}
```

```latex
lambda^new = (gamma_2 + 2 (epsilon + k_{r,0})) k_T / rho_{2,inf}
           = 206.7 rad/s   (gamma_2 = 8)
           = 95.6  rad/s   (gamma_2 = 3)
```

**一层版（`ppc_onelayer_abl.py`），其中 `gamma_1 mu_1 = 0`：**

```latex
edot_1^{diag} = e_2 + xi  ->  e_{2,ss} != 0  (t -> inf)
=> e_1^{diag}(t) = e_{2,ss} t + O(1)
```

### 极点审计（`dt = 2 ms`）

| 通道 | `M_ii` | `lambda_old` | `lambda_old * dt` | 判定 |
|---|---|---|---|---|
| base_x/y | 169.98 | 73.9 | 0.15 | ok |
| yaw | 8.73 | 83.5 | 0.17 | ok |
| j1 | 0.090 | 1063.7 | **2.13** | **失稳** |
| j2 | 2.866 | 104.3 | 0.21 | ok |
| j3 | 0.072 | 1307.1 | **2.61** | **失稳** |
| j4 | 0.641 | 212.1 | 0.42 | ok |
| j5 | 0.040 | 2278.6 | **4.56** | **失稳** |
| j6 | 0.048 | 1917.3 | **3.83** | **失稳** |
| j7 | 0.004 | 25116 | **50.2** | **失稳** |

失稳惯量阈值（`lambda dt = 1`）：

```latex
M* = (2 K_{b,0} k_T / rho_{2,inf})
     / (1/dt - (gamma_2 + 2 epsilon) k_T / rho_{2,inf})
   = 0.208 kg m^2
```

### 域外分流

**`ppc01_baseline.py` / `ppc02_mu_unclip.py`（v0/v1）：** 无域处理，由变换饱和静默吸收越界。

**`ppc03_reinit.py`（v2，重 init 常开）：**

```latex
G_1 : q_d^+ = q,  e_1^+ = 0
G_2 : alpha_f^+ = qdot,  e_2^+ = 0
```

**`ppc04_noclip.py` / `ppc05_fixed.py`（v3/v4，开关）：**

```latex
REINIT = 1 : 同 (64)，附事件流 {(t, ch, max|phi|)}
REINIT = 0 : mu 非实数 => tau_cmd = 0
             qdot_d != 0 => edot_1 > 0 恒立 => 永久失控
```

**`ppc_onelayer_abl.py`（1L，仅 L2）：**

```latex
G_2 : alpha_f^+ = qdot,  e_2^+ = 0   (无 e_1 / G_1 概念)
```

### PPC 差异表

| 特性 | v0 | v1 | v2 | v3 | v4 | 1L |
|---|---|---|---|---|---|---|
| 变换算子 | clip x2 | clip x1 | 纯 | 纯 | 纯 | 纯 |
| 域处理 | 静默 | 静默 | REINIT 固定 | REINIT 开关 | REINIT 开关 | REINIT（L2） |
| 位置环 `e_1 / mu_1` | ✓ | ✓ | ✓ | ✓ | ✓ | ✗（诊断） |
| `gamma_1` | 0.4 | 0.4 | 0.4 | 0.4 | **0.25** | — |
| `gamma_2` | 3.0 | 3.0 | 3.0 | 3.0 | **8.0** | 3.0 |
| 阻尼结构 | `K_b / M_ii` | 同 | 同 | 同 | `k_r` 进 `M` | `K_b / M_ii` |
| `rho_{1,inf}` 臂 | 0.12 | 0.12 | 0.12 | 0.025 | **0.04** | 0.12（诊断） |
| `rho_{2,inf}` 臂 | 0.25 | 0.25 | 0.25 | 0.10 | 0.10 | 0.25 |
| 层数 | 2 | 2 | 2 | 2 | 2 | **1** |

---

## 力矩接口分配

### 臂力矩接口（全版本一致）

```latex
tau[4:11] = tau_sigma[3:10]
```

### 底盘力矩接口

**`ppc01_baseline.py` / `ppc02_mu_unclip.py` / `ppc03_reinit.py` / `ppc_onelayer_abl.py`（clip ON）：**

```latex
tau_w = clip( T_f R_{w2b} tau_sigma[0:3], +-50 )
```

**`ppc04_noclip.py` / `ppc05_fixed.py`（无 clip，仅记录）：**

```latex
tau_w = T_f R_{w2b} tau_sigma[0:3]
```

### 输出映射

**clip ON 组：**

```latex
tau_cmd = clip(tau, tau_lo, tau_hi)
```

**无 clip 组：**

```latex
tau_cmd = tau
```

饱和诊断（仅测量）：

```latex
n_i = sum_t 1{ tau_i > tau_bar_i + 1e-9  or  tau_i < -tau_bar_i - 1e-9 }
r_i = n_i / N_steps
```

```latex
tau_bar = [50, 50, 50, 50, 87, 87, 100, 87, 12, 12, 12]
```

### 镇定段

**均匀 PD（v0–v3，1L）：**

```latex
tau_{a,i} = clip( h_i + 50 e_hold,i - 10 qdot_i, tau_lo, tau_hi )
lambda dt = 10 dt / M_ii
```

> j7（`M_77 = 0.004`）给出 `lambda dt = 5 > 2`；`z = 1 - 5 = -4` 每步反号 → 0.05 s 处 NaN → P2 违例 → `t = 0` 事件级联。

**M 加权（v4）：**

```latex
tau_{a,i} = h_i + M_ii (25 e_hold,i - 10 qdot_i)
lambda dt = 0.02   (与 M_ii 无关)
```

### 力矩接口差异表

| 特性 | v0 | v1 | v2 | v3 | v4 | 1L |
|---|---|---|---|---|---|---|
| 臂力矩映射 | 直通 | 直通 | 直通 | 直通 | 直通 | 直通 |
| 底盘力分配 | `T_f R_{w2b}` | 同 | 同 | 同 | 同 | 同 |
| 轮矩 clip | ±50 | ±50 | ±50 | **无** | **无** | ±50 |
| 输出 clip | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ |
| 超限计数 | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ |
| Settle 结构 | 均匀 PD | 均匀 PD | 均匀 PD | 均匀 PD | **M 加权** | 均匀 PD |
| DSC 基线类 | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |

### 数据流总图

```latex
p_d, Q* --[轨迹生成]--> DLS --[零空间]--> [限位回避: v0 无] --> Slew --> qdot_des
       --[PPC]--> 积分器 q_d --> e_1 --> mu_1
              --> alpha = qdot_des - gamma_1 mu_1
                  [1L: alpha = qdot_des +- K_I int(e_2)]
              --> 滤波 lambda_f = 20 --> alpha_f --> e_2 --> mu_2
              --> tau_sigma = h + M (alphadot_f - gamma_2 mu_2 - epsilon s_2)
                              - { K_b s_2 | v4: M k_r s_2 }
       --[力矩接口]--> 力分配 --> tau --> [输出 clip: v3/v4 无]
              --> [域外: v0/v1 静默 | v2+ REINIT/NaN 开关]
```

---

## 数值实现

**公共。** 显式 Euler，`alpha_f` 先更新后积分；NaN 守卫在 `q, qdot, tau` 非有限时强制 `tau_cmd = 0`；前提检查 (P1)–(P4)：

```latex
alpha_f^+ = alpha_f + dt alphadot_f
```

```latex
(P1) e_1(0) = 0
(P2) |qdot_i(0)| < 0.9 rho_{2,i}(0)
(P3) ||qdot_des|| 有界
(P4) M 对角占优
```

P3 由分通道饱和与 Slew 保证。P4 之外的非对角耦合并入集中扰动 `d`。

---

## 诊断量

公共指标：

```latex
over_i = max_t ( |e_{1,i}(t)| - 0.9 rho_{1,i}(t) )
|e_1|_{ss,i} = mean_{T_end - 5 <= t <= T_end} |e_{1,i}(t)|
```

| 版本 | 新增诊断 |
|---|---|
| v1 | 首次引入上述指标 |
| v2 | 事件流 `{(t, ch, max||phi||)}`；P1/P2 打印 |
| v3 | `n_i`、`r_i`；NaN 事件流；REINIT vs 裸 PPC 对照 |
| v4 | 通过 `audit_closed_loop` 逐通道在线计算旧/新极点、失稳阈值、gamma_2 窗口、位置环 PM、屏障平衡点；逐通道 L2 事件计数与 `max_t phi` |
| 1L | 漂移斜率 `s_hat_i`（判据 `s_hat_i ~ e_{2,ss,i}`）；`mean(|e_2|)(t)`；INJECT 恒等式验证 |

---

## 符号—代码映射

| 公式 | 代码位置（除注明外六版同名） |
|---|---|
| 差速底盘 | `_base_output` / 常量 `WHEEL_R, HALF_AXLE, HALF_TRACK` |
| 末端雅可比 / 姿态误差 | `world_jacobian` / `orientation_error_world` |
| 动力学 / 集中扰动 | `compute_ctrl`（`h` 与扰动归集） |
| 力分配 | `T_FORCE` 常量 / `_base_output` |
| 轨迹生成 / IK 层 | `generate_trajectory` / `v_cmd, w_cmd` |
| DLS / 自适应阻尼 | `manipulability` / `adaptive_damping2` |
| Slew | `SlewLimiter.__call__` |
| 零空间投影 / 梯度 / 门控 | `pinv(rcond=1e-6)` / `manipulability_gradient` |
| 限位排斥 / 刹车 | `null_space_limit_repulsion` / `apply_limit_braking`（v0 无） |
| 漏斗 | `PPFVector.f` |
| 变换（逐版不同） | `PPFVector.transform` |
| 命令滤波 | `compute_ctrl` 滤波段 |
| Level-1 / Level-2 | `alpha = ...`、`tau_sigma = ...` |
| 域外分流 | `domain_ok` 分支 / `compute_ctrl` 尾段 |
| 力矩接口 / 镇定段 | `_base_output` / `settle_step` |
| 审计（仅 v4） | `audit_closed_loop` |

---

## 附录：DSC 对照基线

仅存在于 `ppc01_baseline.py` / `ppc02_mu_unclip.py` / `ppc03_reinit.py`。通过 `USE_PPC=False` 启用。作为「无预设性能」的 A/B 参照。

```latex
tau_virt = (C qdot + G - tau_passive)
         + M ( alphadot + k_d (qdot_des - qdot) )
alphadot = -lambda_f (alpha - qdot_des),  k_d = 5
```

输出同样经 ctrlrange clip。在 `ppc04_noclip.py`、`ppc05_fixed.py`、`ppc_onelayer_abl.py` 中已移除。

---

## 设计总结：七个替换对象

| # | 对象 | v0 | v1 | v2 | v3 | v4 | 1L |
|---|---|---|---|---|---|---|---|
| 1 | 变换算子 | clip x2 | clip x1 | 纯 | 纯 | 纯 | 纯 |
| 2 | 域处理 | 静默 | 静默 | 重 init 固定 | 重 init 开关 | 重 init 开关 | 重 init（L2） |
| 3 | 阻尼结构 | `K_b / M_ii` | 同 | 同 | 同 | `k_r` 进 `M` | `K_b / M_ii` |
| 4 | 漏斗宽度 | 宽 | 宽 | 宽 | 窄 | 窄 + 重平衡 | 宽 |
| 5 | `gamma_2` | 3 | 3 | 3 | 3 | **8** | 3 |
| 6 | Settle 极点 | `k / M_ii` | 同 | 同 | 同 | 加速度级 | `k / M_ii` |
| 7 | 层数 | 2 | 2 | 2 | 2 | 2 | **1** |

**叙事主线。** v0/v1 违反 `T' >= k_T` → 屏障增益被钳死 → v2 恢复前提，但旧律遗留 `88.9 / M_ii` 通道失配 → v3 收窄 `rho_2`，使灵敏度 `lambda dt ~ 1 / (rho_2 M_ii)` 加倍，暴露出 `M* = 0.208` 覆盖 `{j1, j3, j5, j6, j7}` → v4 统一极点，由平衡窗口设定 `gamma_2 = 8` → 1L 以漂移定理收尾：位置层不可去除，而 INJECT 恒等式表明「修好的一层律」其实就是两层律的变装。
