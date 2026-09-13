# 代码说明
覆盖：`PPF.py`(v0)、`PPF2.py`(v1)、`PPC.py`(v2)、`ppcnoclip.py`(v3)、`PPCnoclip2.py`(v4)、`PPComlyVol.py`(1L 消融)。**全版本一致的公式正常编号书写；存在差异的条目按“文件名 + <实现>”分列。**
## 版本总览
| 文件 | 角色 | 一句话本质 |
|---|---|---|
| `PPF.py` | v0 基线 | 变换双重饱和 + 输出限幅 ON + 无限位回避 |
| `PPF2.py` | v1 | μ 去饱和 [A] + 首次引入限位排斥/刹车 [B] |
| `PPC.py` | v2 | 纯变换 + 域外重init [A+]（固定开）+ P1/P2 前提检查 |
| `ppcnoclip.py` | v3 | 去输出限幅 [X2] + 漏斗收紧 + REINIT/裸PPC 双档 [X3] |
| `PPCnoclip2.py` | v4 | 极点修复 [FIX1] + 屏障强度 [FIXB] + M加权settle [FIXA] + 审计 |
| `PPComlyVol.py` | 1L 旁支 | 仅速度层消融；限幅/宽漏斗刻意回到基线设定，唯一变量=层数 |
---
## 0. 系统模型
$$
\dot x_e = J_e(q)\,\dot q,\\
\ddot q = M^{-1}(q)\big(B(q)\,\tau - h(q,\dot q) + d(t)\big),\\
h = C\dot q + G - \tau_{\text{passive}} \tag{0.1}
$$
$h$ 由 `qfrc_bias − qfrc_passive` 获得；$d$ 集中全部未建模项。符号：$q\in\mathbb{R}^{10}$，$\tau\in\mathbb{R}^{11}$。
---
## 1. 任务层
**1.1–1.9 全版本一致**（期望轨迹、世界系姿态误差双覆盖、分通道向量饱和 $\mathrm{sat}_{0.6}/\mathrm{sat}_{1.0}$、可操作度 $w=\sqrt{\det(J_pJ_p^\top)}$、自适应阻尼 $\lambda^2$、DLS 解 $\dot q_{task}$、零空间投影 $N$、有限差分梯度奇异规避 + 门控）：
$$
v_c=\mathrm{sat}_{0.6}(\dot p_d + 10e_p),\quad
\omega_c=\mathrm{sat}_{1.0}(\omega_d + 2e_R),\quad
\dot q_{task}=J_e^\top(J_eJ_e^\top+\lambda^2 I_6)^{-1}\begin{bmatrix}v_c\\\omega_c\end{bmatrix} \tag{1.1}
$$
**1.10 限位回避（存在性差异）：**
**PPF.py**
```
<无此模块: q̇_raw = q̇_task + q̇_null^(w)
 (无排斥 v_rep、无刹车 — v1 起才引入 [B])>
```
**PPF2.py / PPC.py / ppcnoclip.py / PPCnoclip2.py / PPComlyVol.py**
```
<排斥: v_rep,j = k_lim·s² 1{距限位<d_m}, d_m=0.15, k_lim=2
 刹车: 距限位<d_b=0.05 时禁止朝壁速度分量(半平面投影)
 q̇_raw = Brake( q̇_task + q̇_null^(w) + N·[0₃; v_rep] )>
```
**1.11 Slew 限变化（全版本一致，$S=2$）：**
$$
\dot q_{des}=\dot q_{des}^{-}+\min\Big(1,\ \frac{S\,\Delta t}{\|\dot q_{raw}-\dot q_{des}^{-}\|_\infty}\Big)(\dot q_{raw}-\dot q_{des}^{-}) \tag{1.2}
$$
---
## 2. 参考积分器与误差定义
**PPF.py / PPF2.py / PPC.py / ppcnoclip.py / PPCnoclip2.py**（两层结构公共）：
```
<q_d(0)=q(0) 精确 (P1);  q̇_d = q̇_des;  e₁ = q − q_d (yaw 分量 wrap)>
```
**PPComlyVol.py**
```
<控制回路内无 q_d、无 e₁ (纯前馈结构)
 诊断积分器(主循环内, 不进任何控制量): q_d_diag ← ∫q̇_des
 e₁_diag = q − q_d_diag   (仅用于末 20s 漂移斜率拟合报告)>
```
---
## 3. 漏斗函数与变换（核心差异区）
**漏斗形状（全版本一致，$\alpha‘=2$，$T=10$）：**
$$
\rho(t)=(\rho_0-\rho_\infty)\exp\!\Big(-\frac{\alpha’ t}{T-t}\Big)+\rho_\infty \tag{3.1}
$$
$$
\rho_{1,0}=[0.20,0.20,0.30\,|\,0.25{\times}7],\qquad
\rho_{2,0}=[2.5,2.5,2.0\,|\,3.0{\times}7] \tag{3.2}
$$
**漏斗终值 $\rho_\infty$（差异）：**
**PPF.py / PPF2.py / PPC.py**
```
<ρ₁∞ = [0.15,0.15,0.20 | 0.12×7];  ρ₂∞ = [0.20,0.20,0.20 | 0.25×7]>
```
**ppcnoclip.py**
```
<ρ₁∞ = [0.05,0.05,0.05 | 0.025×7];  ρ₂∞ = [0.10,0.10,0.10 | 0.10×7]  (收紧暴露振荡)>
```
**PPCnoclip2.py**
```
<ρ₁∞ = [0.05,0.05,0.05 | 0.04×7]  [FIX2: 臂位置漏斗 0.025→0.04]
 ρ₂∞ = [0.10,0.10,0.10 | 0.10×7]>
```
**PPComlyVol.py**
```
<ρ₂∞ = [0.20,0.20,0.20 | 0.25×7]  (回到宽漏斗)
 ρ₁_diag∞ = [0.15,0.15,0.20 | 0.12×7]  (仅画参考线: "两层版会承诺什么")>
```
**变换实现（差异）：**
**PPF.py**
```
<φ = clip(e/ρ, −0.899, 0.899)
 μ = clip( ln((0.9+φ)/(0.9−φ)), −1, 1 )          [S2 双重饱和]
 有效屏障增益: |φ|≤0.416 内 dμ/de = T'(φ)/ρ;
 |φ|>0.416 处 μ≡±1 ⇒ dμ/de = 0
 (漏斗外半段增量增益被完全切断, 越界静默发生且无记录)>
```
**PPF2.py**
```
<φ = clip(e/ρ, −0.899, 0.899)   (内层数值保险保留)
 μ = ln((0.9+φ)/(0.9−φ))        ([A] 外层 clip 删除)
 μ 事实上限 = ln(0.9·1.799 / 0.9·0.001) ≈ ±7.5>
```
**PPC.py / ppcnoclip.py / PPCnoclip2.py / PPComlyVol.py**
```
<φ = e/ρ,  μ = ln((0.9+φ)/(0.9−φ))    (纯微分同胚, 零截断)
 前提监测 domain_ok: |e| < 0.9ρ(t);  越界分流见 §7>
```
**公共性质**：$\partial\mu/\partial\varphi=\frac{1.8}{0.81-\varphi^2}$（边界 $\to\infty$，屏障本体）；反函数 $e=0.9\rho\tanh(\mu/2)$。
---
## 4. 控制律（差异集中区）
**命令滤波器（全版本一致）**：$\dot\alpha_f=-\lambda_f(\alpha_f-\alpha)$，$\lambda_f=20$，$\alpha_f(0)=0$；$e_2=\dot q-\alpha_f$。
**4.1 Level-1 虚拟控制：**
**PPF.py / PPF2.py / PPC.py / ppcnoclip.py**
```
<α = q̇_des − γ₁μ₁,  γ₁ = 0.4>
```
**PPCnoclip2.py**
```
<α = q̇_des − γ₁μ₁,  γ₁ = 0.25   [FIX2: 位置环PM 44°→51°]>
```
**PPComlyVol.py**
```
<INJECT_INTEGRAL=False (默认): α = q̇_des            (纯前馈, 无位置反馈)
 INJECT_INTEGRAL=True:  α = q̇_des − K_I·z,  K_I=2,  ż = e₂,  |z|≤0.5
 (z=∫e₂ 即 e₁ 的变装; 漂移消失 ⟺ 验证"修好的一层 ≡ 两层变装")>
```
**4.2 Level-2 力矩律：**
**PPF.py / PPF2.py / PPC.py / ppcnoclip.py / PPComlyVol.py**
```
<K_b = 2 + 0.02·min(‖q̇‖,100)²,  γ₂ = 3.0,  ε = 0.15,  δ = 0.5
 τ_σ = h + M(α̇_f − γ₂μ₂ − ε·s₂) − K_b·s₂,   s₂ = tanh(μ₂/δ)
 闭环: Mė₂ = −γ₂Mμ₂ − (εM + K_b I₁₀)s₂ + d
 等效极点 λ_i = (γ₂+2ε)k_T/ρ₂ + 2K_b·k_T/(ρ₂·M_ii)
              ↑乘M项通道一致    ↑裸力矩项÷本关节惯量 → 通道失配源>
```
**PPCnoclip2.py（DAMP_IN_M=True，默认新律）**
```
<k_r = 0.5 + 0.005·min(‖q̇‖,100)²,  γ₂ = 8.0        [FIX1][FIXB]
 τ_σ = h + M(α̇_f − γ₂μ₂ − ε·s₂ − k_r·s₂)
 闭环: Mė₂ = −γ₂Mμ₂ − (ε+k_r)M·s₂ + d
 等效极点 λ = (γ₂ + 2(ε+k_r))·k_T/ρ₂   — 全通道一致, 与 M_ii 解耦>
```
**PPCnoclip2.py（DAMP_IN_M=False，旧律 A/B 对照档）**
```
<K_b = 2 + 0.02‖q̇‖²,  γ₂ = 8.0
 τ_σ = h + M(α̇_f − γ₂μ₂ − ε·s₂) − K_b·s₂
 (与上五版同构, 仅 γ₂=8; 用于复现 j1/j3/j5–j7 振荡)>
```
**闭环误差动力学（两层版公共）**：$\dot e_1=e_2+\tilde\alpha-\gamma_1\mu_1$，$\dot{\tilde\alpha}=-\lambda_f\tilde\alpha-\dot\alpha$。
**PPComlyVol.py**
```
<ė₁_diag = q̇ − q̇_des = e₂ + (α_f − q̇_des)
 α_f 对 q̇_des 是一阶跟踪(无偏) ⇒ 稳态 ė₁_diag ≈ e₂,ss ≠ 0
 ⇒ 漂移定理: 位置误差末段线性漂移, 斜率 ≈ e₂,ss — 一层架构唯一无法承诺的量
 Lyapunov 仅 V₂ = ½‖μ₂‖² (速度漏斗); e₁ 无任何保证>
```
两层版 Lyapunov：$V=\tfrac12\|\mu_1\|^2+\tfrac12\|\mu_2\|^2$，漏斗不变集 $|e_1|\le0.9\rho_1\tanh(\bar\chi/2\gamma_1)$。
---
## 5. 底盘力分配
**(5.1) 几何与 (5.2)–(5.3) 两模式公式全版本一致**（$T_f=\frac r4 K\,\mathrm{diag}(1,1,\frac1{L^2})$，$J_w=\frac1r K$，$r=0.12$，$L=0.427$；$T_f^\top T_f=\frac{r^2}{4}I_3$）。**轮矩 clip 差异：**
**PPF.py / PPF2.py / PPC.py / PPComlyVol.py**
```
<τ_w = clip( T_f R_z^⊤(ψ) τ_σ[0:3], ±50 )     (力分配模式)
 τ_w = clip( 4(J_w u_body − ω_w), ±50 )        (轮速伺服模式)>
```
**ppcnoclip.py / PPCnoclip2.py**
```
<同式, 无 clip [X2]>
```
---
## 6. 输出映射
**PPF.py / PPF2.py / PPC.py / PPComlyVol.py**
```
<τ_cmd = clip(τ, τ_lo, τ_hi)>
```
**ppcnoclip.py / PPCnoclip2.py**
```
<τ_cmd = τ    (无任何 clip; 超限诊断 n_exceed,i = Σₜ 1{τ_i 超旧上限 τ̄}, 记录但不动手)
 物理现实: MuJoCo ctrllimited 内部钳制不可删 (本模型实测 0/11 执行器 limited)>
```
---
## 7. 定义域维持机制
**PPF.py / PPF2.py**
```
<无域处理: 越界由 §3 的 φ/μ 饱和静默吸收
 (漏斗可被无声穿越, 无事件、无记录 — v0/v1 被替代的直接原因)>
```
**PPC.py**
```
<REINIT 固定启用 ([A+] 机制, 无开关), 跳变非限幅:
 L1: |e₁|≥0.9ρ₁ ⇒ 记录事件(t,通道,φ),  q_d⁺=q,  e₁⁺=0
 L2: |e₂|≥0.9ρ₂ ⇒ 记录事件,            α_f⁺=q̇,  e₂⁺=0
 reset_states 打印 P1/P2 前提校验>
```
**ppcnoclip.py / PPCnoclip2.py**
```
<REINIT_ON 开关 [X3]:
 True  — 同 PPC.py (跳变投影, 事件全记录)
 False — 裸 PPC: 无域处理, μ 非有限 ⇒ 记录NaN事件, τ_cmd=0
         (q_d 继续积分, e₁ 单调增大, 永久失控)
 两档对照 ⟹ 证明"零事件 ⟹ 纯PPC独立收敛"与"最小机制必要性">
```
**PPComlyVol.py**
```
<REINIT 固定启用, 仅 L2 一级: |e₂|≥0.9ρ₂ ⇒ α_f⁺=q̇, e₂⁺=0>
```
---
## 8. 数值实现
**公共**：显式 Euler（$\alpha_f$ 先算后积分）；NaN 守卫（$q,\dot q,\tau$ 非有限 $\Rightarrow\tau_{cmd}=0$，故障隔离非控制环节）；交接前提 (P1) $e_1(0)=0$ / (P2) $|\dot q_i(0)|<0.9\rho_{2,i}(0)$（PPC.py 起打印）。
**Settling 段（差异）：**
**PPF.py / PPF2.py / PPC.py / ppcnoclip.py / PPComlyVol.py**
```
<τ_a = clip( h + 50·e_hold − 10·q̇, τ_lo, τ_hi )    (均匀PD)
 隐含极点 kp/M_ii, kd/M_ii — 在 M₇₇=0.004 时 λdt=5, 逐拍反号发散>
```
**PPCnoclip2.py**
```
<τ_a = h + M_ii·(25·e_hold − 10·q̇)    [FIXA] M加权: 极点=加速度级(25,10)
 λdt ≤ 0.02 与惯量无关;  LIMIT_TORQUE=False ⇒ 无 clip>
```
**终局诊断（逐版增量）：**
| 版本 | 诊断量 |
|---|---|
| `PPF2.py` | +[C] 终局漏斗违规报告 $\text{over}_i,\ |e_1|_{ss}$ |
| `PPC.py` | +重init事件流（(t, 通道， max\|φ\|)） |
| `ppcnoclip.py` | +超限计数 (6.2) + NaN 事件流 |
| `PPCnoclip2.py` | +`audit_closed_loop`（逐通道新旧极点 λdt / 失稳阈值 $M^*$ / $k_r$ 可行域 / 位置环 PM / 屏障平衡点 $\varphi^*$ 预估）+ L2 逐通道统计 |
| `PPComlyVol.py` | +末 20s 漂移斜率拟合（斜率≈$e_{2,ss}$?）+ mean\|e₂\| 曲线 + INJECT 验证注释 |
---
## 9. 差异总表
| 特性 | PPF | PPF2 | PPC | ppcnoclip | PPCnoclip2 | PPComlyVol |
|---|---|---|---|---|---|---|
| 限位回避 [B] | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 位置环 $e_1/\mu_1/\gamma_1$ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗（仅诊断） |
| 变换饱和 | φ+μ 双重 | φ-clip | ✗ | ✗ | ✗ | ✗ |
| 域外处理 | 静默 | 静默 | 重init固定 | 开关/NaN | 开关/NaN | 重init固定(L2) |
| $K$ 项结构 | 裸力矩 | 裸力矩 | 裸力矩 | 裸力矩 | **进 M**（可A/B） | 裸力矩 |
| $\gamma_1/\gamma_2$ | 0.4/3.0 | 0.4/3.0 | 0.4/3.0 | 0.4/3.0 | **0.25/8.0** | —/3.0 |
| 输出 clip | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ |
| $\rho_{1,\infty}^{\text{arm}}$ | 0.12 | 0.12 | 0.12 | 0.025 | **0.04** | 0.12（诊断） |
| $\rho_{2,\infty}^{\text{arm}}$ | 0.25 | 0.25 | 0.25 | 0.10 | 0.10 | 0.25 |
| settle | 均匀PD | 均匀PD | 均匀PD | 均匀PD | **M加权** | 均匀PD |
| DSC 基线类 | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
**数据流标注版**（括号内 = 差异所属版本）：
```
p_d,Q* → DLS+零空间 → [限位回避: v0无] → Slew → q̇_des ─┬→ 积分器 q_d → e₁ → μ₁ ─┐
                      (PPF.py 无 B 模块)               │  [1L: 仅诊断,不进控制]   │
                                                       └→ α = q̇_des − γ₁μ₁ [1L: α=q̇_des ± K_I∫e₂]
                                                            → 滤波 λ_f=20 → α_f → e₂ → μ₂
                                                            → τ_σ = h+M(α̇_f−γ₂μ₂−εs₂) −{K_b·s₂ | v4新律: −M·k_r s₂}
                                                            → 力分配 → τ → [输出clip: v3/v4无]
                                                            → [域外: v0/v1静默饱和 | v2+重init/NaN开关]
```
---
## 附录 A：DSC 对照基线（仅 PPF.py / PPF2.py / PPC.py 含，`USE_PPC=False`）
$$
\tau_{virt}=(C\dot q+G-\tau_{passive})+M\big(\dot\alpha+k_d(\dot q_{des}-\dot q)\big),\quad
\dot\alpha=-\lambda_f(\alpha-\dot q_{des}),\quad k_d=5 \tag{A.1}
$$
输出同样经 ctrlrange clip；无 PPF、无屏障，作为“无预设性能”的 A/B 参照。v3 起删除该类。
---
**两点使用说明**：① 六版共享 §0、§1.1–1.9、滤波器、§5 公式与全部上游模块（`orientation_error_world`/DLS/雅可比/`SlewLimiter`），代码层面约 70% 逐字重复；② 理论立场上，§3 的变换与 §7 的域处理是**同一件事的两半**——纯变换承诺“域内增益→∞ 的屏障”，域处理承诺“变换永不在域外求值”；v0/v1 用饱和同时牺牲了这两半（屏障增益被钳死 + 越界静默），v2 起才把两者都还给定理。
