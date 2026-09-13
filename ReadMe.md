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

\[
q=\begin{bmatrix}x_b\\ y_b\\ \psi\\ q_{a,1}\\ \vdots\\ q_{a,7}\end{bmatrix}\in\mathbb{R}^{10},\qquad
\tau=\begin{bmatrix}\tau_{w,1}\\ \vdots\\ \tau_{w,4}\\ \tau_{a,1}\\ \vdots\\ \tau_{a,7}\end{bmatrix}\in\mathbb{R}^{11},\qquad
\zeta=\begin{bmatrix}\dot{x}_b\\ \dot{y}_b\\ \dot{\psi}\end{bmatrix}\in\mathbb{R}^{3} \tag{1}
\]

关节限位 $q_{a,j}\in[\ell_j,\,h_j]$。$J^{+}$ 为截断伪逆（$rcond=10^{-6}$），$\mathbf{1}\{\cdot\}$ 为示性函数。

### 差速底盘

\[
R_z(\psi)=\begin{bmatrix}\cos\psi&-\sin\psi&0\\ \sin\psi&\cos\psi&0\\ 0&0&1\end{bmatrix},\quad
R_{w2b}=R_z^\top(\psi),\quad
u_{body}=R_{w2b}\,\zeta \tag{2}
\]

\[
r=0.12,\quad L=0.2225+0.2045=0.427,\quad
K=\begin{bmatrix}1&1&L\\ 1&-1&-L\\ 1&-1&L\\ 1&1&-L\end{bmatrix},\quad
\omega_w=J_w\,u_{body},\quad J_w=\frac{1}{r}K \tag{3}
\]

### 末端雅可比

\[
\dot{x}_e=\begin{bmatrix}\dot{p}_{ee}\\ \omega_{ee}\end{bmatrix}=J_e(q)\dot{q},\quad
J_e=[\,J_b\mid J_a\,]\in\mathbb{R}^{6\times 10} \tag{4}
\]

\[
J_b^{(\mathrm{lin})}=[\,e_x\ \ e_y\ \ \hat{z}\times(p_{ee}-p_b)\,],\quad
J_b^{(\mathrm{ang})}=[\,\mathbf{0}_{2\times 3}\ \ \hat{z}\,]^\top \tag{5}
\]

\[
J_a^{(\mathrm{lin})}[\cdot,i]=z_i\times(p_{ee}-p_i),\quad
J_a^{(\mathrm{ang})}[\cdot,i]=z_i,\quad
J_p=J_e^{[1:3,:]},\ J_r=J_e^{[4:6,:]} \tag{6}
\]

### 姿态误差（世界系，最短路径）

\[
q_{err}=Q^{*}\otimes Q_{ee}^{-1}=(\eta,\boldsymbol{\varepsilon}),\qquad
q_{err}\leftarrow-q_{err}\ \ \text{if}\ \eta<0 \tag{7}
\]

\[
e_R=\theta\,\hat{\boldsymbol{\varepsilon}},\quad
\theta=2\arctan2(\|\boldsymbol{\varepsilon}\|,\eta),\quad
\hat{\boldsymbol{\varepsilon}}=\frac{\boldsymbol{\varepsilon}}{\|\boldsymbol{\varepsilon}\|}\quad(\|\boldsymbol{\varepsilon}\|=0\Rightarrow e_R=0) \tag{8}
\]

### 动力学

\[
M(q)\ddot{q}+C(q,\dot{q})\dot{q}+G(q)=B(q)\tau+\tau_{passive}(q,\dot{q})+d(t) \tag{9}
\]

\[
\ddot{q}=M^{-1}(B\tau-h+d),\qquad h\triangleq C\dot{q}+G-\tau_{passive} \tag{10}
\]

$h$ 逐 DOF 读自 `qfrc_bias − qfrc_passive`。集中扰动为

\[
d=\underbrace{-J_e^\top F_{ext}}_{\text{接触}}
+\underbrace{\tau_{fric}}_{\text{摩擦}}
+\underbrace{(I-BB^{+})B\tau}_{\text{平面解耦残差}}
+\underbrace{\tau_{sat}}_{\text{饱和}}
+\underbrace{O(\Delta t)}_{\text{离散化}} \tag{11}
\]

### 执行器映射

\[
B=\begin{bmatrix}T_f R_{w2b}&\mathbf{0}_{3\times 7}\\ \mathbf{0}_{7\times 4}&I_7\end{bmatrix},\quad
T_f=\frac{r}{4}K\,\mathrm{diag}\Big(1,1,\tfrac{1}{L^2}\Big) \tag{12}
\]

\[
\tau_{phys}=\mathrm{sat}_{[\text{ctrlrange}]}(\tau_{cmd})\quad(\text{仅 ctrllimited 执行器；本模型 }0/11) \tag{13}
\]

力分配精确性——由 $K^\top K=4\,\mathrm{diag}(1,1,L^2)$：

\[
\frac{1}{r}K^\top T_f=I_3
\quad\Longrightarrow\quad
R_z(\psi)\,\tfrac{1}{r}K^\top\tau_w=\tau_{\sigma,[0:3]} \tag{14}
\]

---

## 轨迹生成

### 参考轨迹（全版本一致）

\[
p_d(t)=\begin{bmatrix}\cos(0.2t)\\ \sin(0.2t)\\ 0.7\end{bmatrix},\quad
\dot{p}_d=0.2\begin{bmatrix}-\sin(0.2t)\\ \cos(0.2t)\\ 0\end{bmatrix},\quad
Q^{*}=[0,1,0,0]^\top,\quad \omega_d\equiv 0 \tag{15}
\]

\[
e_p=p_d-p_{ee},\qquad e_R\ \text{由 (8) 给出} \tag{16}
\]

### 分通道向量饱和（全版本一致）

\[
\mathrm{sat}_a(x)=\frac{a}{\max(\|x\|,a)}\,x,\quad
v_c=\mathrm{sat}_{0.6}(\dot{p}_d+10e_p),\quad
\omega_c=\mathrm{sat}_{1.0}(\omega_d+2e_R) \tag{17}
\]

### 阻尼最小二乘主任务（全版本一致）

\[
\dot{q}_{task}=\arg\min_{\dot{q}}\ \Big\|J_e\dot{q}-\begin{bmatrix}v_c\\ \omega_c\end{bmatrix}\Big\|^2+\lambda^2\|\dot{q}\|^2
\ \Longrightarrow\
\dot{q}_{task}=J_e^\top(J_eJ_e^\top+\lambda^2 I_6)^{-1}\begin{bmatrix}v_c\\ \omega_c\end{bmatrix} \tag{18}
\]

\[
w(q)=\sqrt{\det(J_pJ_p^\top)},\quad
\lambda^2=\lambda_{\min}^2+\lambda_0^2\Big(1-\min\big(\tfrac{w}{w_{thr}},1\big)\Big)^2 \tag{19}
\]

其中 $\lambda_{\min}=10^{-3}$，$\lambda_0=0.05$，$w_{thr}=0.02$。

### Slew 限变化（全版本一致，$S=2$）

\[
\dot{q}_{des}(t)=\dot{q}_{des}(t-\Delta t)+\min\Big(1,\ \frac{S\Delta t}{\|\dot{q}_{raw}-\dot{q}_{des}(t-\Delta t)\|_\infty}\Big)\big(\dot{q}_{raw}-\dot{q}_{des}(t-\Delta t)\big) \tag{20}
\]

---

## 零空间参考信号

优先级：**任务 1**（末端跟踪，主任务）→ **任务 2**（奇异规避）→ **任务 3**（限位回避）。任务 2 与任务 3 通过任务 1 的零空间投影。

### 零空间投影算子（全版本一致）

\[
N=I_{10}-J_e^{+}J_e,\qquad J_e N=0\ (\text{精确}),\qquad N^2=N \tag{21}
\]

### 任务 2 —— 奇异规避（全版本一致）

\[
g_{w,k}=\frac{w(q+\varepsilon_g e_k)-w(q)}{\varepsilon_g},\quad \varepsilon_g=10^{-5},\quad
\dot{q}_{null}^{(w)}=k_{eff}\,N\,\frac{g_w}{\|g_w\|} \tag{22}
\]

\[
k_{eff}=0.5\,\mathrm{clip}\Big(\frac{0.5-\|e_p\|}{0.3},0,1\Big),\quad
\text{启用}\iff k_{eff}>10^{-6}\wedge w>0.3\,w_{thr}\wedge\|g_w\|>10^{-12} \tag{23}
\]

### 任务 3 —— 限位回避

**`ppc01_baseline.py`（v0）：** 该模块不存在。

\[
\dot{q}_{null}^{(\lim)}\equiv\mathbf{0}_7 \tag{24}
\]

**`ppc02_mu_unclip.py` / `ppc03_reinit.py` / `ppc04_noclip.py` / `ppc05_fixed.py` / `ppc_onelayer_abl.py`（v1+）：** 排斥 + 刹车。

\[
s_j^-=\frac{d_m-(q_{a,j}-\ell_j)}{d_m},\quad
s_j^+=\frac{d_m-(h_j-q_{a,j})}{d_m},\quad d_m=0.15,\ k_{\lim}=2 \tag{25}
\]

\[
v_{rep,j}=k_{\lim}(s_j^-)^2\mathbf{1}\{q_{a,j}-\ell_j<d_m\}
-k_{\lim}(s_j^+)^2\mathbf{1}\{h_j-q_{a,j}<d_m\} \tag{26}
\]

\[
(P\dot{q})_{3+j}=\dot{q}_{3+j}\cdot\mathbf{1}\Big\{\neg(h_j-q_{a,j}<d_b\wedge\dot{q}_{3+j}>0)
\wedge\neg(q_{a,j}-\ell_j<d_b\wedge\dot{q}_{3+j}<0)\Big\},\quad d_b=0.05 \tag{27}
\]

### 合成

**`ppc01_baseline.py`（v0）：**

\[
\dot{q}_{raw}=\dot{q}_{task}+\dot{q}_{null}^{(w)} \tag{28}
\]

**`ppc02_mu_unclip.py` / `ppc03_reinit.py` / `ppc04_noclip.py` / `ppc05_fixed.py` / `ppc_onelayer_abl.py`（v1+）：**

\[
\dot{q}_{raw}=P\big(\dot{q}_{task}+\dot{q}_{null}^{(w)}+\dot{q}_{null}^{(\lim)}\big) \tag{29}
\]

> **注。** 刹车算子作用于零空间投影之后，故仅当刹车激活时 $J_e P\dot{q}_{null}\neq 0$。残差并入 (11) 的 $d$。

### 差异表

| 特性 | v0 | v1 | v2 | v3 | v4 | 1L |
|---|---|---|---|---|---|---|
| 任务 1（末端跟踪） | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 任务 2（奇异规避） | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 任务 3（限位回避） | **✗** | ✓ | ✓ | ✓ | ✓ | ✓ |
| 合成 | $\dot{q}_{task}+\dot{q}_{null}^{(w)}$ | $P(\cdots+v_{rep})$ | 同 v1 | 同 v1 | 同 v1 | 同 v1 |

---

## 预设性能控制

### 参考积分器与误差定义

**`ppc01_baseline.py` / `ppc02_mu_unclip.py` / `ppc03_reinit.py` / `ppc04_noclip.py` / `ppc05_fixed.py`（两层版）：**

\[
q_d(0)=q(0)\ \text{精确 (P1)},\quad \dot{q}_d=\dot{q}_{des},\quad e_1=q-q_d,\quad e_{1,2}\leftarrow\mathrm{wrap}(e_{1,2}) \tag{30}
\]

\[
\mathrm{wrap}(a)=(a+\pi)\bmod 2\pi-\pi \tag{31}
\]

**`ppc_onelayer_abl.py`（1L）：** 控制回路内无 $q_d$、$e_1$、$\mu_1$。保留诊断积分器但从不反馈：

\[
q_d^{diag}\leftarrow\int\dot{q}_{des},\qquad e_1^{diag}=q-q_d^{diag} \tag{32}
\]

精确关系：$e_1^{diag}(t)=\int_0^t(\dot{q}-\dot{q}_{des})\,d\tau=\int_0^t(e_2+\xi)\,d\tau$，其中 $\xi\triangleq\alpha_f-\alpha$。

### 漏斗函数

\[
\rho(t)=(\rho_0-\rho_\infty)\exp\!\Big(-\frac{\alpha' t}{T-t}\Big)+\rho_\infty,\quad \alpha'=2,\ T=10\ \mathrm{s} \tag{33}
\]

初值（全版本一致）：

\[
\rho_{1,0}=[0.20,0.20,0.30\mid 0.25^{\times 7}],\quad
\rho_{2,0}=[2.5,2.5,2.0\mid 3.0^{\times 7}] \tag{34}
\]

终值（差异）：

**`ppc01_baseline.py` / `ppc02_mu_unclip.py` / `ppc03_reinit.py`：**

\[
\rho_{1,\infty}=[0.15,0.15,0.20\mid 0.12^{\times 7}],\quad
\rho_{2,\infty}=[0.20,0.20,0.20\mid 0.25^{\times 7}] \tag{35}
\]

**`ppc04_noclip.py`：**

\[
\rho_{1,\infty}=[0.05,0.05,0.05\mid 0.025^{\times 7}],\quad
\rho_{2,\infty}=[0.10,0.10,0.10\mid 0.10^{\times 7}] \tag{36}
\]

**`ppc05_fixed.py`：**

\[
\rho_{1,\infty}=[0.05,0.05,0.05\mid 0.04^{\times 7}],\quad
\rho_{2,\infty}=[0.10,0.10,0.10\mid 0.10^{\times 7}] \tag{37}
\]

**`ppc_onelayer_abl.py`：**

\[
\rho_{2,\infty}=[0.20,0.20,0.20\mid 0.25^{\times 7}],\quad
\rho_{1,\infty}^{diag}=[0.15,0.15,0.20\mid 0.12^{\times 7}]\ \text{（仅作绘图参考线）} \tag{38}
\]

### 理想变换（v2–v4 与 1L 实现的数学对象）

\[
\varphi=\frac{e}{\rho(t)},\quad
T(\varphi)=\ln\frac{0.9+\varphi}{0.9-\varphi}=2\,\mathrm{artanh}\Big(\frac{\varphi}{0.9}\Big) \tag{39}
\]

\[
\mathcal{D}_t=\{e:|e|<0.9\rho(t)\},\quad T:\mathcal{D}_t\to\mathbb{R}\ \text{严格微分同胚} \tag{40}
\]

\[
T'(\varphi)=\frac{1.8}{0.81-\varphi^2}\ \geq\ k_T\triangleq T'(0)=\frac{1.8}{0.81}=2.2222,\quad
\lim_{\varphi\to\pm 0.9}T'(\varphi)=+\infty \tag{41}
\]

\[
e=0.9\,\rho\tanh(\mu/2),\qquad \mathrm{sign}(e)=\mathrm{sign}(\mu) \tag{42}
\]

### 各版本实际变换

**`ppc01_baseline.py`（v0，双重饱和）：**

\[
\varphi_c=\mathrm{clip}(e/\rho,-0.899,0.899),\quad
\mu=\mathrm{clip}(T(\varphi_c),-1,1) \tag{43}
\]

\[
\frac{d\mu}{de}=\begin{cases}T'(\varphi)/\rho, & |\varphi|\le 0.9\tanh\tfrac{1}{2}=0.4159\\[4pt] 0, & 0.4159<|\varphi|<0.9\end{cases} \tag{44}
\]

**`ppc02_mu_unclip.py`（v1）：**

\[
\varphi_c=\mathrm{clip}(e/\rho,-0.899,0.899),\quad \mu=T(\varphi_c),\quad |\mu|\le T(0.899)=\ln 1799\approx 7.5 \tag{45}
\]

**`ppc03_reinit.py` / `ppc04_noclip.py` / `ppc05_fixed.py` / `ppc_onelayer_abl.py`（v2+）：**

\[
\mu=T(e/\rho)\quad\text{（零截断，(39)–(42) 全部成立）} \tag{46}
\]

### Level-1 虚拟控制

**`ppc01_baseline.py` / `ppc02_mu_unclip.py` / `ppc03_reinit.py` / `ppc04_noclip.py`：**

\[
\alpha=\dot{q}_{des}-\gamma_1\mu_1,\qquad \gamma_1=0.4 \tag{47}
\]

**`ppc05_fixed.py`：**

\[
\alpha=\dot{q}_{des}-\gamma_1\mu_1,\qquad \gamma_1=0.25 \tag{48}
\]

**`ppc_onelayer_abl.py`：**

\[
\text{INJECT}=0:\ \alpha=\dot{q}_{des};\qquad
\text{INJECT}=1:\ \alpha=\dot{q}_{des}-K_I z,\quad K_I=2,\quad \dot{z}=e_2,\quad |z|\le 0.5 \tag{49}
\]

> **恒等式。** 当 $z(0)=e_1(0)=0$，有 $z(t)=\int_0^t e_2\,d\tau=e_1(t)$。注入版等价于 $\alpha=\dot{q}_{des}-K_I e_1$——即去除变换的两层律（$\gamma_1\mapsto K_I$）。

### 命令滤波器（全版本一致）

\[
\dot{\alpha}_f=-\lambda_f(\alpha_f-\alpha),\quad \lambda_f=20,\quad \alpha_f(0)=0;\qquad \dot{\xi}=-\lambda_f\xi-\dot{\alpha} \tag{50}
\]

\[
e_2=\dot{q}-\alpha_f \tag{51}
\]

### Level-2 力矩律

**旧律（`ppc01_baseline.py` / `ppc02_mu_unclip.py` / `ppc03_reinit.py` / `ppc04_noclip.py` / `ppc_onelayer_abl.py`）：**

\[
s_2=\tanh(\mu_2/\delta),\quad \delta=0.5,\quad
K_b=2+0.02\min(\|\dot{q}\|,100)^2,\quad \gamma_2=3.0,\ \varepsilon=0.15 \tag{52}
\]

\[
\tau_\sigma=h+M(\dot{\alpha}_f-\gamma_2\mu_2-\varepsilon s_2)-K_b\,s_2 \tag{53}
\]

**新律（`ppc05_fixed.py`，DAMP_IN_M=True，默认）：**

\[
k_r=0.5+0.005\min(\|\dot{q}\|,100)^2,\quad \gamma_2=8.0,\ \varepsilon=0.15 \tag{54}
\]

\[
\tau_\sigma=h+M(\dot{\alpha}_f-\gamma_2\mu_2-\varepsilon s_2-k_r s_2) \tag{55}
\]

**A/B 对照律（`ppc05_fixed.py`，DAMP_IN_M=False）：**

\[
K_b=2+0.02\min(\|\dot{q}\|,100)^2,\ \gamma_2=8.0;\quad
\tau_\sigma=h+M(\dot{\alpha}_f-\gamma_2\mu_2-\varepsilon s_2)-K_b\,s_2 \tag{56}
\]

### 闭环误差动力学

位置层（两层版公共）：

\[
\dot{e}_1=e_2+\xi-\gamma_1\mu_1,\qquad
\dot{\mu}_1=T'(\varphi_1)\Big(\frac{\dot{e}_1}{\rho_1}-\varphi_1\sigma_1\Big) \tag{57}
\]

速度层：

\[
\text{旧律:}\quad \dot{e}_{2,i}=-\gamma_2\mu_{2,i}-(\varepsilon+K_b/M_{ii})\tanh\frac{\mu_{2,i}}{\delta}+\frac{d_i}{M_{ii}} \tag{58}
\]

\[
\text{新律:}\quad \dot{e}_{2,i}=-\gamma_2\mu_{2,i}-(\varepsilon+k_r)\tanh\frac{\mu_{2,i}}{\delta}+\frac{d_i}{M_{ii}} \tag{59}
\]

线性化极点（$\mu_2\to 0$，$T'\to k_T$，$\tfrac{d}{d\mu}\tanh(\mu/\delta)|_0=2/\delta$）：

\[
\lambda_i^{old}=\underbrace{(\gamma_2+2\varepsilon)\frac{k_T}{\rho_{2,\infty}}}_{\text{经 }M,\ \text{通道一致}}
+\underbrace{\frac{2K_{b,0}\,k_T}{\rho_{2,\infty}M_{ii}}}_{\text{裸 }K_b,\ \div M_{ii}}
=\ 73.3+\frac{88.9}{M_{ii}} \tag{60}
\]

\[
\lambda^{new}=(\gamma_2+2(\varepsilon+k_{r,0}))\frac{k_T}{\rho_{2,\infty}}
=\begin{cases}206.7\ \mathrm{rad/s}, & \gamma_2=8\\ 95.6\ \mathrm{rad/s}, & \gamma_2=3\end{cases} \tag{61}
\]

**一层版（`ppc_onelayer_abl.py`），其中 $\gamma_1\mu_1\equiv 0$：**

\[
\dot{e}_1^{diag}=e_2+\xi\ \xrightarrow{t\to\infty}\ e_{2,ss}\neq 0
\quad\Longrightarrow\quad
e_1^{diag}(t)=e_{2,ss}\,t+O(1) \tag{62}
\]

### 极点审计（$\Delta t=2$ ms）

| 通道 | $M_{ii}$ | $\lambda^{old}$ | $\lambda^{old}\Delta t$ | 判定 |
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

失稳惯量阈值（$\lambda\Delta t=1$）：

\[
M^{*}=\frac{2K_{b,0}k_T/\rho_{2,\infty}}{1/\Delta t-(\gamma_2+2\varepsilon)k_T/\rho_{2,\infty}}=0.208\ \mathrm{kg\,m^2} \tag{63}
\]

### 域外分流

**`ppc01_baseline.py` / `ppc02_mu_unclip.py`（v0/v1）：** 无域处理，由 §"各版本实际变换" 中的饱和静默吸收越界。

**`ppc03_reinit.py`（v2，重 init 常开）：**

\[
\mathcal{G}_1:\ q_d^{+}=q,\ e_1^{+}=0;\qquad
\mathcal{G}_2:\ \alpha_f^{+}=\dot{q},\ e_2^{+}=0 \tag{64}
\]

**`ppc04_noclip.py` / `ppc05_fixed.py`（v3/v4，开关）：**

\[
\text{REINIT}=1:\ \text{同 (64)，附事件流 } \{(t,\text{ch},\max|\varphi|)\} \tag{65}
\]

\[
\text{REINIT}=0:\ \mu\notin\mathbb{R}\ \Rightarrow\ \tau_{cmd}=0;\quad
\dot{q}_d\neq 0\ \Rightarrow\ \dot{e}_1>0\ \text{恒立}\ \Rightarrow\ \text{永久失控} \tag{66}
\]

**`ppc_onelayer_abl.py`（1L，仅 L2）：**

\[
\mathcal{G}_2:\ \alpha_f^{+}=\dot{q},\ e_2^{+}=0\quad(\text{无 }e_1/\mathcal{G}_1\text{ 概念}) \tag{67}
\]

### PPC 差异表

| 特性 | v0 | v1 | v2 | v3 | v4 | 1L |
|---|---|---|---|---|---|---|
| 变换算子 | clip×2 | clip×1 | 纯 | 纯 | 纯 | 纯 |
| 域处理 | 静默 | 静默 | REINIT 固定 | REINIT 开关 | REINIT 开关 | REINIT（L2） |
| 位置环 $e_1/\mu_1$ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗（诊断） |
| γ₁ | 0.4 | 0.4 | 0.4 | 0.4 | **0.25** | — |
| γ₂ | 3.0 | 3.0 | 3.0 | 3.0 | **8.0** | 3.0 |
| 阻尼结构 | $K_b/M_{ii}$ | 同 | 同 | 同 | $k_r$ 进 $M$ | $K_b/M_{ii}$ |
| ρ₁∞ 臂 | 0.12 | 0.12 | 0.12 | 0.025 | **0.04** | 0.12（诊断） |
| ρ₂∞ 臂 | 0.25 | 0.25 | 0.25 | 0.10 | 0.10 | 0.25 |
| 层数 | 2 | 2 | 2 | 2 | 2 | **1** |

---

## 力矩接口分配

### 臂力矩接口（全版本一致）

\[
\tau_{[4:11]}=\tau_{\sigma,[3:10]} \tag{68}
\]

### 底盘力矩接口

**`ppc01_baseline.py` / `ppc02_mu_unclip.py` / `ppc03_reinit.py` / `ppc_onelayer_abl.py`（clip ON）：**

\[
\tau_w=\mathrm{clip}\big(T_f R_{w2b}\,\tau_{\sigma,[0:3]},\pm 50\big) \tag{69}
\]

**`ppc04_noclip.py` / `ppc05_fixed.py`（无 clip，仅记录）：**

\[
\tau_w=T_f R_{w2b}\,\tau_{\sigma,[0:3]} \tag{70}
\]

### 输出映射

**clip ON 组：**

\[
\tau_{cmd}=\mathrm{clip}(\tau,\tau_{lo},\tau_{hi}) \tag{71}
\]

**无 clip 组：**

\[
\tau_{cmd}=\tau \tag{72}
\]

饱和诊断（仅测量）：

\[
n_i=\sum_t \mathbf{1}\{\tau_i>\bar{\tau}_i+10^{-9}\vee\tau_i<-\bar{\tau}_i-10^{-9}\},\quad r_i=\frac{n_i}{N_{steps}} \tag{73}
\]

\[
\bar{\tau}=[50,50,50,50,87,87,100,87,12,12,12] \tag{74}
\]

### 镇定段

**均匀 PD（v0–v3，1L）：**

\[
\tau_{a,i}=\mathrm{clip}(h_i+50\,e_{hold,i}-10\,\dot{q}_i,\tau_{lo},\tau_{hi}),\quad \lambda\Delta t=\frac{10\Delta t}{M_{ii}} \tag{75}
\]

> j7（$M_{77}=0.004$）给出 $\lambda\Delta t = 5 > 2$；$z=1-5=-4$ 每步反号 → 0.05 s 处 NaN → P2 违例 → $t=0$ 事件级联。

**M 加权（v4）：**

\[
\tau_{a,i}=h_i+M_{ii}(25\,e_{hold,i}-10\,\dot{q}_i),\quad \lambda\Delta t=0.02\ (\text{与 }M_{ii}\text{ 无关}) \tag{76}
\]

### 力矩接口差异表

| 特性 | v0 | v1 | v2 | v3 | v4 | 1L |
|---|---|---|---|---|---|---|
| 臂力矩映射 | 直通 | 直通 | 直通 | 直通 | 直通 | 直通 |
| 底盘力分配 | $T_f R_{w2b}$ | 同 | 同 | 同 | 同 | 同 |
| 轮矩 clip | ±50 | ±50 | ±50 | **无** | **无** | ±50 |
| 输出 clip | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ |
| 超限计数 | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ |
| Settle 结构 | 均匀 PD | 均匀 PD | 均匀 PD | 均匀 PD | **M 加权** | 均匀 PD |
| DSC 基线类 | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |

### 数据流总图

\[
\begin{aligned}
p_d,Q^* &\xrightarrow{\S\text{轨迹生成}} \text{DLS} \xrightarrow{\S\text{零空间}} [\text{限位回避：v0 无}] \to \text{Slew} \to \dot{q}_{des} \\
&\xrightarrow{\S\text{PPC}} \text{积分器}\ q_d \to e_1 \to \mu_1 \to \alpha = \dot{q}_{des} - \gamma_1\mu_1\ [\text{1L: }\alpha=\dot{q}_{des}\pm K_I\!\int e_2] \\
&\quad \to \text{滤波}\ \lambda_f=20 \to \alpha_f \to e_2 \to \mu_2 \\
&\quad \to \tau_\sigma = h+M(\dot{\alpha}_f-\gamma_2\mu_2-\varepsilon s_2)-\{K_b s_2\mid \text{v4: }-M k_r s_2\} \\
&\xrightarrow{\S\text{力矩接口}} \text{力分配} \to \tau \to [\text{输出 clip：v3/v4 无}] \\
&\quad \to [\text{域外：v0/v1 静默}\mid \text{v2+ REINIT/NaN 开关}]
\end{aligned}
\]

---

## 数值实现

**公共。** 显式 Euler，$\alpha_f$ 先更新后积分；NaN 守卫在 $q,\dot{q},\tau$ 非有限时强制 $\tau_{cmd}=0$；前提检查 (P1)–(P4)：

\[
\alpha_f^{+}=\alpha_f+\Delta t\,\dot{\alpha}_f \tag{77}
\]

\[
(P1)\ e_1(0)=0;\quad (P2)\ |\dot{q}_i(0)|<0.9\rho_{2,i}(0);\quad (P3)\ \|\dot{q}_{des}\|\ \text{有界};\quad (P4)\ M\ \text{对角占优} \tag{78}
\]

P3 由 (17) 与 (20) 保证。P4 之外的非对角耦合并入 (11)。

---

## 诊断量

公共指标：

\[
\mathrm{over}_i=\max_t\big(|e_{1,i}(t)|-0.9\rho_{1,i}(t)\big),\quad
|e_1|_{ss,i}=\operatorname*{mean}_{T_{end}-5\le t\le T_{end}}|e_{1,i}(t)| \tag{79}
\]

| 版本 | 新增诊断 |
|---|---|
| v1 | 首次引入 (79) |
| v2 | 事件流 $\{(t,\text{ch},\max\|\varphi\|)\}$；P1/P2 打印 |
| v3 | $n_i$、$r_i$；NaN 事件流；REINIT vs 裸 PPC 对照 |
| v4 | 通过 `audit_closed_loop` 逐通道在线计算 (60)–(63)、(65)–(66)；逐通道 L2 事件计数与 $\max_t\varphi$ |
| 1L | 漂移斜率 $\hat{s}_i$（判据 $\hat{s}_i\approx e_{2,ss,i}$）；$\overline{\|e_2\|}(t)$；INJECT 恒等式验证 |

---

## 符号—代码映射

| 公式 | 代码位置（除注明外六版同名） |
|---|---|
| (2)–(3) | `_base_output` / 常量 `WHEEL_R, HALF_AXLE, HALF_TRACK` |
| (4)–(8) | `world_jacobian` / `orientation_error_world` |
| (9)–(11) | `compute_ctrl`（`h` 与集中扰动） |
| (12)–(14) | `T_FORCE` 常量 / `_base_output` |
| (15)–(17) | `generate_trajectory` / IK 层 `v_cmd, w_cmd` |
| (18)–(19) | `manipulability` / `adaptive_damping2` / DLS |
| (20) | `SlewLimiter.__call__` |
| (21)–(23) | `pinv(rcond=1e-6)` / `manipulability_gradient` / 门控 |
| (25)–(27) | `null_space_limit_repulsion` / `apply_limit_braking`（v0 无） |
| (33)–(38) | `PPFVector.f` |
| (39)–(46) | `PPFVector.transform`（逐版不同） |
| (50)–(51) | `compute_ctrl` 滤波段 |
| (52)–(56) | `alpha = ...`、`tau_sigma = ...` |
| (64)–(67) | `domain_ok` 分支 / `compute_ctrl` 尾段 |
| (68)–(76) | `_base_output` / `settle_step` |
| (60)–(63) | `audit_closed_loop`（仅 v4） |

---

