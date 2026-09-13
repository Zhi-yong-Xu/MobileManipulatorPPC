#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[1L] 一层 PPC (仅速度层) 对照实验
==========================================================================
结构: τ = h + M(α̇_f − γ₂μ₂ − ε·tanh(μ₂/δ)) − K_b·tanh(μ₂/δ)
      α = q̇_des (纯前馈, 无位置反馈项 γ₁μ₁)
      无参考积分器进控制律, 无 e₁, 无位置漏斗
保留: 速度层PPF(纯变换) / 命令滤波 / 鲁棒项 / 上游参考生成
      (DLS+零空间+刹车+Slew+任务饱和) / 输出限幅(默认恢复, 隔离变量)
诊断: q_d_diag = ∫q̇_des (不进控制), e1_diag = q − q_d_diag
      → 预期: e2 进漏斗(小), e1_diag 线性漂移(速率≈e2_ss)
开关: INJECT_INTEGRAL  False=纯一层(默认)
      True=α 加 −K_I∫e₂ — 预言漂移消失(∫e₂ 即 e₁, "修好的一层"≡两层变装)
==========================================================================
"""
import numpy as np
import mujoco
import mujoco.viewer

# ============================================================
# 0. 配置
# ============================================================
XML_PATH = r'models\summit_panda\summit_xls_merged copy.xml'
ARM_JOINT_NAMES = ['panda_joint1', 'panda_joint2', 'panda_joint3',
                   'panda_joint4', 'panda_joint5', 'panda_joint6',
                   'panda_joint7']
EE_BODY_NAME   = 'panda_hand'
BASE_BODY_NAME = 'base_footprint'

T_SETTLE, T_SIM = 1.0, 40.0
REC_SKIP, PRT_SKIP = 10, 1000

# ---- 本实验开关 ----
LIMIT_TORQUE   = True     # 恢复输出限幅 (与[A+]基线一致, 隔离"层数"变量)
INJECT_INTEGRAL = False   # False=纯一层; True=注入∫e₂ (验证"≡两层变装")
KI             = 2.0      # 注入积分增益
INJ_CLIP       = 0.5      # 注入积分状态的安全界(仅此诊断开关用)
REINIT_ON      = True     # L2 越域事件触发重初始化(应全程0事件)
ROBUST_ON      = True
WHEEL_SERVO    = False
SLEW_S         = 2.0

# ---- 速度层 PPC 参数 (与两层版逐字相同) ----
GAMMA2   = 3.0
EPSILON  = 0.15
DELTA    = 0.5
LAMBDA_F = 20.0
KB0, KB1 = 2.0, 0.02

# ---- [B] 限位回避 (上游, 保留) ----
LIM_MARGIN_REP   = 0.15
K_LIM            = 2.0
LIM_MARGIN_BRAKE = 0.05

# ============================================================
# 1. 轨迹
# ============================================================
CENTER = np.array([0.0, 0.0, 0.7])
RADIUS, OMEGA = 1.0, 0.2
TARGET_QUAT = np.array([0, 1, 0, 0])

def generate_trajectory(t):
    th = OMEGA * t
    pos = np.array([CENTER[0] + RADIUS*np.cos(th),
                    CENTER[1] + RADIUS*np.sin(th), CENTER[2]])
    vel = np.array([-RADIUS*OMEGA*np.sin(th),
                     RADIUS*OMEGA*np.cos(th), 0.0])
    return pos, TARGET_QUAT.copy(), vel, np.zeros(3)

# ============================================================
# 1b. 姿态误差 + 限位工具 (上游, 保留)
# ============================================================
def orientation_error_world(tq, qc):
    qc_inv = np.zeros(4)
    mujoco.mju_negQuat(qc_inv, np.asarray(qc, float))
    qe = np.zeros(4)
    mujoco.mju_mulQuat(qe, np.asarray(tq, float), qc_inv)
    if qe[0] < 0.0:
        qe = -qe
    w = np.zeros(3)
    mujoco.mju_quat2Vel(w, qe, 1.0)
    return w

_arm_limit_cache = {}
def _arm_joint_limit_info(model):
    key = id(model)
    if key not in _arm_limit_cache:
        info = []
        for n in ARM_JOINT_NAMES:
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            info.append((model.jnt_qposadr[j], bool(model.jnt_limited[j]),
                         float(model.jnt_range[j][0]),
                         float(model.jnt_range[j][1])))
        _arm_limit_cache[key] = info
    return _arm_limit_cache[key]

def null_space_limit_repulsion(model, data,
                               dm=LIM_MARGIN_REP, k_lim=K_LIM):
    v = np.zeros(len(ARM_JOINT_NAMES))
    n_act = 0
    for i, (adr, lim, lo, hi) in enumerate(_arm_joint_limit_info(model)):
        if not lim:
            continue
        q = data.qpos[adr]
        if (q - lo) < dm:
            s = (dm - (q - lo)) / dm
            v[i] += k_lim * s * s
            n_act += 1
        if (hi - q) < dm:
            s = (dm - (hi - q)) / dm
            v[i] -= k_lim * s * s
            n_act += 1
    return v, n_act

def apply_limit_braking(model, data, dq, dm=LIM_MARGIN_BRAKE):
    for i, (adr, lim, lo, hi) in enumerate(_arm_joint_limit_info(model)):
        if not lim:
            continue
        q = data.qpos[adr]
        if (hi - q) < dm and dq[3+i] > 0.0:
            dq[3+i] = 0.0
        if (q - lo) < dm and dq[3+i] < 0.0:
            dq[3+i] = 0.0
    return dq

def joint_limit_margin(model, data, arm_qpos_adrs, arm_joint_ids):
    worst, wj = 1e9, -1
    for i, jid in enumerate(arm_joint_ids):
        if not model.jnt_limited[jid]:
            continue
        q = data.qpos[arm_qpos_adrs[i]]
        lo, hi = model.jnt_range[jid]
        m = min(q - lo, hi - q)
        if m < worst:
            worst, wj = m, i
    return worst, wj

# ============================================================
# 2. PPF —— 纯变换 (速度层用; ρ1 仅作诊断参考线)
# ============================================================
class PPFVector:
    def __init__(self, f0, f_inf, alpha, T, eps_lo=-0.9, eps_up=0.9):
        self.f0    = np.atleast_1d(f0).astype(float)
        self.f_inf = np.atleast_1d(f_inf).astype(float)
        self.alpha, self.T = alpha, T
        self.eps_lo, self.eps_up = eps_lo, eps_up

    def f(self, t):
        if t < self.T:
            return (self.f0 - self.f_inf) * np.exp(
                -self.alpha * t / (self.T - t)) + self.f_inf
        return self.f_inf.copy()

    def domain_ok(self, e, t):
        return np.abs(e) < self.eps_up * self.f(t)

    def transform(self, e, t):
        ft = self.f(t)
        phi = e / ft
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.log((self.eps_up * (phi - self.eps_lo)) /
                          (-self.eps_lo * (self.eps_up - phi)))

def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi

class SlewLimiter:
    def __init__(self, n, S=2.0):
        self.prev = np.zeros(n)
        self.S = S
    def __call__(self, dq_new, dt):
        dq_new = np.asarray(dq_new, float)
        d = dq_new - self.prev
        slope = np.max(np.abs(d)) / dt
        if slope > self.S:
            dq_new = self.prev + d * (self.S / slope)
        self.prev = dq_new.copy()
        return dq_new

# ============================================================
# 3. 底盘层公共参数
# ============================================================
WHEEL_R, HALF_AXLE, HALF_TRACK = 0.12, 0.2225, 0.2045
_L = HALF_AXLE + HALF_TRACK
_K = np.array([[ 1,  1, _L], [ 1, -1, -_L],
               [ 1, -1, _L], [ 1,  1, -_L]])
J_WHEELS = (1/WHEEL_R) * _K
T_FORCE  = (WHEEL_R/4.0) * _K @ np.diag([1, 1, 1/(_L*_L)])
KP_W     = 4.0

PANDA_TAU = [87, 87, 100, 87, 12, 12, 12]

def make_torque_limits(model, n_arm):
    lo = np.zeros(4 + n_arm); hi = np.zeros(4 + n_arm)
    default = [50]*4 + PANDA_TAU
    for k in range(4 + n_arm):
        if model.actuator_ctrllimited[k]:
            lo[k] = model.actuator_ctrlrange[k, 0]
            hi[k] = model.actuator_ctrlrange[k, 1]
        else:
            lo[k], hi[k] = -default[k], default[k]
    return lo, hi

# ============================================================
# 4. [1L] 一层 PPC 控制器: 只有速度层
# ============================================================
class MobileManipulatorPPC1LController:
    """速度层PPC: e2 = q̇ − α_f, α = q̇_des (无位置反馈, 无参考积分器进控制)"""

    def __init__(self, model, arm_joint_names,
                 gamma2=GAMMA2, epsilon=EPSILON,
                 filter_gain=LAMBDA_F, kb0=KB0, kb1=KB1):
        self.model = model
        self.num_arm  = len(arm_joint_names)
        self.num_ctrl = 3 + self.num_arm
        self.num_out  = 4 + self.num_arm

        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                    BASE_BODY_NAME)
        self.base_body_id = base_id
        bj   = model.body_jntadr[base_id]
        badr = model.jnt_dofadr[bj]
        bqa  = model.jnt_qposadr[bj]
        self.base_dof_adrs = [badr + 0, badr + 1, badr + 5]
        self.base_qpos_xy  = [bqa + 0, bqa + 1]
        self.arm_dof_adrs, self.arm_qpos_adrs = [], []
        for n in arm_joint_names:
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            self.arm_dof_adrs.append(model.jnt_dofadr[j])
            self.arm_qpos_adrs.append(model.jnt_qposadr[j])
        self.all_dof_adrs = self.base_dof_adrs + self.arm_dof_adrs
        self.M_full = np.zeros((model.nv, model.nv))

        self.J_wheels = J_WHEELS
        self.T_force  = T_FORCE
        self.wheel_dof_adrs = [model.jnt_dofadr[model.actuator_trnid[i, 0]]
                               for i in range(4)]
        self.kp_w = KP_W

        self.gamma2, self.epsilon = gamma2, epsilon
        self.filter_gain = filter_gain
        self.kb0, self.kb1 = kb0, kb1

        self.ctrl_lo, self.ctrl_hi = make_torque_limits(model, self.num_arm)

        # 速度漏斗 (与两层版逐字相同)
        f0_2  = np.array([2.5, 2.5, 2.0] + [3.0]*self.num_arm)
        fin_2 = np.array([0.20, 0.20, 0.20] + [0.25]*self.num_arm)
        self.ppf2 = PPFVector(f0_2, fin_2, 2.0, 10.0)
        # 位置漏斗仅作诊断参考线 ("两层版会承诺什么")
        f0_1  = np.array([0.20, 0.20, 0.30] + [0.25]*self.num_arm)
        fin_1 = np.array([0.15, 0.15, 0.20] + [0.12]*self.num_arm)
        self.ppf1_diag = PPFVector(f0_1, fin_1, 2.0, 10.0)

        self.dq_des = np.zeros(self.num_ctrl)
        self.alpha_f, self.alpha_f_dot = None, None
        self.e2_int = np.zeros(self.num_ctrl)     # 仅 INJECT_INTEGRAL 用
        self.t = 0.0
        self.mu2_last = np.zeros(self.num_ctrl)

        self.reinit_events = []
        self.nan_events    = []

    def set_target(self, dq_des):
        if len(dq_des) == self.num_ctrl:
            self.dq_des = np.array(dq_des)
        else:
            raise ValueError(f"期望维度 {self.num_ctrl}, 实际 {len(dq_des)}")

    def reset_states(self, data, dq_des=None):
        if dq_des is not None:
            self.dq_des = np.array(dq_des)
        self.alpha_f = np.zeros(self.num_ctrl)
        self.alpha_f_dot = np.zeros(self.num_ctrl)
        self.e2_int[:] = 0.0
        self.t = 0.0
        self.reinit_events, self.nan_events = [], []

        dq0 = np.array([data.qvel[a] for a in self.all_dof_adrs])
        lim0 = 0.9 * self.ppf2.f(0.0)
        bad = np.where(np.abs(dq0) >= lim0)[0]
        if len(bad):
            print(f'[WARN] 前提P2违例 @handover: channels={bad.tolist()}')
        else:
            print(f'[PREMISE] P2\': max|dq0|={np.max(np.abs(dq0)):.3f} < '
                  f'{np.min(lim0):.3f}  OK  (一层无P1/e1(0)概念)')

    def _get_q(self, data):
        q = np.zeros(self.num_ctrl)
        q[0] = data.qpos[self.base_qpos_xy[0]]
        q[1] = data.qpos[self.base_qpos_xy[1]]
        xm = data.xmat[self.base_body_id].reshape(3, 3)
        q[2] = np.arctan2(xm[1, 0], xm[0, 0])
        for i, a in enumerate(self.arm_qpos_adrs):
            q[3+i] = data.qpos[a]
        return q

    def settle_step(self, data, q_hold, kp=50.0, kd=10.0):
        tau = np.zeros(self.num_out)
        for i, a in enumerate(self.arm_dof_adrs):
            h_i = data.qfrc_bias[a] - data.qfrc_passive[a]
            tau[4+i] = h_i + kp*(q_hold[i] - data.qpos[a]) \
                            + kd*(0.0 - data.qvel[a])
        if LIMIT_TORQUE:
            return np.clip(tau, self.ctrl_lo, self.ctrl_hi)
        return tau

    def compute_ctrl(self, data, dt):
        q  = self._get_q(data)
        dq = np.array([data.qvel[a] for a in self.all_dof_adrs])
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(dq))):
            return np.zeros(self.num_out)

        # ---- [1L] 虚拟控制: 纯前馈 (无 -γ₁μ₁, 无 q_d, 无 e₁) ----
        alpha = self.dq_des.copy()
        if INJECT_INTEGRAL:
            alpha = alpha - KI * self.e2_int

        self.alpha_f_dot = -self.filter_gain * (self.alpha_f - alpha)
        e2 = dq - self.alpha_f

        # ---- L2 定义域处理 ----
        ok2 = self.ppf2.domain_ok(e2, self.t)
        if not np.all(ok2):
            v = np.where(~ok2)[0]
            phi = np.abs(e2) / self.ppf2.f(self.t)
            if REINIT_ON:
                self.reinit_events.append(
                    (self.t, v.copy(), float(phi[v].max())))
                print(f'[REINIT] t={self.t:.2f}s L2 ch={v.tolist()} '
                      f'max|phi|={phi[v].max():.3f} -> alpha_f<-dq')
                self.alpha_f = dq.copy()
                e2 = np.zeros_like(e2)

        mu2 = self.ppf2.transform(e2, self.t)
        self.mu2_last = mu2

        if not np.all(np.isfinite(mu2)):
            self.nan_events.append((self.t, np.where(~ok2)[0].copy()))
            print(f'[NaN-FALLBACK] t={self.t:.2f}s L2 out-of-domain -> '
                  f'zero torque')
            self.t += dt
            return np.zeros(self.num_out)

        mujoco.mj_fullM(self.model, data, self.M_full)
        idx = np.array(self.all_dof_adrs)
        M = self.M_full[np.ix_(idx, idx)]
        h = np.array([data.qfrc_bias[a] - data.qfrc_passive[a]
                      for a in self.all_dof_adrs])

        vnorm = min(np.linalg.norm(dq), 100.0)
        Kb = self.kb0 + self.kb1 * vnorm ** 2

        if ROBUST_ON:
            s2s = np.tanh(mu2 / DELTA)
            tau_sigma = h + M @ (self.alpha_f_dot - self.gamma2*mu2
                                 - self.epsilon*s2s) - Kb * s2s
        else:
            tau_sigma = h + M @ (self.alpha_f_dot - self.gamma2*mu2)

        # ---- 注入积分 (诊断开关; ∫e₂ 即 e₁ 的变装) ----
        if INJECT_INTEGRAL:
            self.e2_int += dt * e2
            self.e2_int = np.clip(self.e2_int, -INJ_CLIP, INJ_CLIP)

        self.alpha_f += self.alpha_f_dot * dt
        self.t += dt

        tau = self._base_output(data, tau_sigma)
        if not np.all(np.isfinite(tau)):
            return np.zeros_like(tau)
        if LIMIT_TORQUE:
            return np.clip(tau, self.ctrl_lo, self.ctrl_hi)
        return tau

    def _base_output(self, data, tau_sigma):
        xm = data.xmat[self.base_body_id].reshape(3, 3)
        yaw = np.arctan2(xm[1, 0], xm[0, 0])
        R_w2b = np.array([[ np.cos(yaw), np.sin(yaw), 0],
                          [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        if WHEEL_SERVO:
            u_body = R_w2b @ self.dq_des[:3]
            w_des  = self.J_wheels @ u_body
            w_act  = np.array([data.qvel[a] for a in self.wheel_dof_adrs])
            tau_w  = self.kp_w * (w_des - w_act)
            if LIMIT_TORQUE:
                tau_w = np.clip(tau_w, -50.0, 50.0)
        else:
            tau_w = self.T_force @ (R_w2b @ tau_sigma[:3])
            if LIMIT_TORQUE:
                tau_w = np.clip(tau_w, -50.0, 50.0)
        return np.concatenate((tau_w, tau_sigma[3:]))

# ============================================================
# 5. 雅可比 + 奇异性 + 零空间 (上游, 保留)
# ============================================================
def _jac_cols(model):
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY_NAME)
    ba = model.jnt_dofadr[model.body_jntadr[b]]
    cols = [ba + 0, ba + 1, ba + 5]
    for n in ARM_JOINT_NAMES:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        cols.append(model.jnt_dofadr[j])
    return np.array(cols)

def world_jacobian(model, data, cols=None):
    ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
    jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, ee)
    if cols is None:
        cols = _jac_cols(model)
    return np.vstack((jacp[:, cols], jacr[:, cols]))

def manipulability(J, with_rotation=False):
    Jp = J if with_rotation else J[:3]
    return float(np.sqrt(max(np.linalg.det(Jp @ Jp.T), 0.0)))

def adaptive_damping2(w, lam0=0.05, w_thr=0.02, lam_min=1e-3):
    r = min(max(w, 0.0) / w_thr, 1.0)
    return lam_min**2 + lam0**2 * (1.0 - r)**2

def manipulability_gradient(model, data, scratch, cols,
                            base_dof_adrs, arm_dof_adrs,
                            eps=1e-5, with_rotation=False):
    w0 = manipulability(world_jacobian(model, data, cols), with_rotation)
    g = np.zeros(3 + len(arm_dof_adrs))
    v = np.zeros(model.nv)
    for k, adr in enumerate([base_dof_adrs[2]] + arm_dof_adrs):
        scratch.qpos[:] = data.qpos
        v[:] = 0.0; v[adr] = 1.0
        mujoco.mj_integratePos(model, scratch.qpos, v, eps)
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        Jk = world_jacobian(model, scratch, cols)
        g[2 + k] = (manipulability(Jk, with_rotation) - w0) / eps
    return w0, g

def null_space_ik_controller(model, data, scratch, cols,
                             tp, tq, tv, two,
                             base_dof_adrs, arm_dof_adrs,
                             kp_p=10.0, kp_r=2.0,
                             v_max=0.6, w_max=1.0,
                             k_null=0.5, w_thr=0.02, lam0=0.05,
                             with_rotation=False):
    ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
    err_p = tp - data.xpos[ee].copy()
    err_r = orientation_error_world(tq, data.xquat[ee].copy())

    v_cmd = tv + kp_p * err_p
    w_cmd = two + kp_r * err_r
    n = np.linalg.norm(v_cmd)
    if n > v_max: v_cmd *= v_max / n
    n = np.linalg.norm(w_cmd)
    if n > w_max: w_cmd *= w_max / n

    desired = np.concatenate([v_cmd, w_cmd])
    J = world_jacobian(model, data, cols)
    w = manipulability(J, with_rotation)
    lam2 = adaptive_damping2(w, lam0, w_thr)
    dq_task = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), desired)

    N = np.eye(3 + 7) - np.linalg.pinv(J, rcond=1e-6) @ J
    dq_null = np.zeros(3 + 7)

    k_eff = k_null * np.clip((0.5 - float(np.linalg.norm(err_p))) / 0.3,
                             0.0, 1.0)
    if k_eff > 1e-6 and w > 0.3 * w_thr:
        _, g = manipulability_gradient(model, data, scratch, cols,
                                       base_dof_adrs, arm_dof_adrs,
                                       with_rotation=with_rotation)
        gn = np.linalg.norm(g)
        if gn > 1e-12:
            dq_null += k_eff * (N @ (g / gn))

    v_rep, n_rep = null_space_limit_repulsion(model, data)
    if n_rep > 0:
        rep10 = np.zeros(3 + 7)
        rep10[3:] = v_rep
        dq_null += N @ rep10

    dq = dq_task + dq_null
    dq = apply_limit_braking(model, data, dq)

    dq_now = np.array([data.qvel[c] for c in cols])
    omega_real = J[3:] @ dq_now
    ern = float(np.linalg.norm(err_r))
    w_along = float(np.dot(omega_real, err_r / ern)) if ern > 0.05 else 0.0

    diag = {'w_along': w_along, 'n_rep': n_rep}
    return (dq, w, np.sqrt(lam2), float(np.linalg.norm(dq_null)),
            float(np.linalg.norm(err_p)), ern,
            float(np.max(np.abs(dq))), diag)

# ============================================================
# 6. 可视化辅助
# ============================================================
def add_line(scn, p1, p2, radius, rgba):
    if scn.ngeom >= scn.maxgeom: return
    mid, diff = (p1+p2)/2, p2-p1
    L = np.linalg.norm(diff)
    if L < 1e-10: return
    d = diff/L; z = np.array([0, 0, 1])
    if abs(np.dot(d, z)) > 0.999:
        mat = np.eye(3) if np.dot(d, z) > 0 else np.diag([1,-1,-1]).astype(float)
    else:
        v = np.cross(z, d); v /= np.linalg.norm(v)
        c, s = np.dot(z, d), np.sqrt(1-np.dot(z, d)**2)
        mat = np.array([[c+v[0]*v[0]*(1-c), v[0]*v[1]*(1-c)-v[2]*s,
                         v[0]*v[2]*(1-c)+v[1]*s],
                        [v[1]*v[0]*(1-c)+v[2]*s, c+v[1]*v[1]*(1-c),
                         v[1]*v[2]*(1-c)-v[0]*s],
                        [v[2]*v[0]*(1-c)-v[1]*s, v[2]*v[1]*(1-c)+v[0]*s,
                         c+v[2]*v[2]*(1-c)]])
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER,
                        np.array([radius, L/2, 0]), mid, mat.flatten(), rgba)
    scn.ngeom += 1

# ============================================================
# 7. 绘图
# ============================================================
def plot_results(rec, ctrl):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    try:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['mathtext.fontset'] = 'stix'
    except Exception:
        pass
    plt.rcParams['font.size'] = 10

    t  = np.array(rec['t'])
    e1d = np.array(rec['e1_diag']); r1 = np.array(rec['rho1_diag'])
    e2 = np.array(rec['e2']);       r2 = np.array(rec['rho2'])
    q  = np.array(rec['q']);        qd = np.array(rec['q_d_diag'])
    tau = np.array(rec['tau'])
    w   = np.array(rec['w']);  ee = np.array(rec['ee'])
    errp = np.array(rec['errp']); errr = np.array(rec['errr'])
    mu2m = np.array(rec['mu2']); jlm = np.array(rec['jlim'])
    e2m = np.array(rec['e2_mean'])

    lbl = ['base x (m)', 'base y (m)', 'base yaw (rad)'] + \
          [f'panda j{i+1} (rad)' for i in range(ctrl.num_arm)]

    # --- fig1: e1_diag 漂移 (核心图) ---
    fig1, ax1 = plt.subplots(5, 2, figsize=(14, 12))
    for i in range(ctrl.num_ctrl):
        a = ax1[i//2, i%2]
        a.plot(t, e1d[:, i], 'b', lw=1.0, label=r'$e_1^{diag}=q-q_d^{diag}$')
        a.plot(t,  0.9*r1[:, i], 'r--', lw=.8)
        a.plot(t, -0.9*r1[:, i], 'r--', lw=.8,
               label='0.9ρ₁ (two-layer would guarantee)')
        # 末段线性拟合斜率标注
        m = t >= t[-1] - 20.0
        if np.sum(m) > 10:
            sl = np.polyfit(t[m], e1d[m, i], 1)[0]
            a.set_title(f'{lbl[i]}   drift slope={sl:+.4f}/s')
        else:
            a.set_title(lbl[i])
        a.grid(alpha=.3); a.legend(fontsize=8)
    fig1.suptitle('ONE-LAYER PPC: position drift of open-loop integrator '
                  '(diagnostic only, not in control)', fontsize=14)
    fig1.tight_layout()
    fig1.savefig('oneL_drift.png', dpi=300, bbox_inches='tight')
    print('[SAVED] oneL_drift.png')

    # --- fig2: e2 速度漏斗 (应成立) ---
    fig2, ax2 = plt.subplots(5, 2, figsize=(14, 12))
    for i in range(ctrl.num_ctrl):
        a = ax2[i//2, i%2]
        a.plot(t, e2[:, i], 'b', lw=1.0, label=fr'$e_2[{i+1}]$')
        a.plot(t,  0.9*r2[:, i], 'r--', lw=.8)
        a.plot(t, -0.9*r2[:, i], 'r--', lw=.8, label='funnel 0.9ρ₂')
        a.fill_between(t, -0.9*r2[:, i], 0.9*r2[:, i], alpha=.08, color='r')
        a.set_title(lbl[i]); a.grid(alpha=.3); a.legend(fontsize=8)
    fig2.suptitle('ONE-LAYER PPC: velocity funnel (the ONLY guarantee '
                  'this architecture has)', fontsize=14)
    fig2.tight_layout()
    fig2.savefig('oneL_e2funnel.png', dpi=300, bbox_inches='tight')
    print('[SAVED] oneL_e2funnel.png')

    # --- fig3: q vs q_d_diag ---
    fig3, ax3 = plt.subplots(5, 2, figsize=(14, 12))
    for i in range(ctrl.num_ctrl):
        a = ax3[i//2, i%2]
        a.plot(t, qd[:, i], 'r--', lw=1.0, label='$q_d^{diag}$')
        a.plot(t, q[:, i], 'b', lw=1.0, label='$q$')
        a.set_title(lbl[i]); a.grid(alpha=.3); a.legend(fontsize=8)
    fig3.suptitle('ONE-LAYER PPC: actual q vs integrated reference '
                  '(diverging = drift)', fontsize=14)
    fig3.tight_layout()
    fig3.savefig('oneL_tracking.png', dpi=300, bbox_inches='tight')
    print('[SAVED] oneL_tracking.png')

    # --- fig4: 诊断 ---
    fig4, a4 = plt.subplots(1, 6, figsize=(28, 3.5))
    a4[0].plot(t, w, 'b', lw=1.2)
    a4[0].axhline(0.02, color='r', ls='--', lw=.8)
    a4[0].set_title('Manipulability w(t)')
    a4[1].semilogy(t, mu2m, 'b', lw=1.0)
    a4[1].set_title('max|mu2| (pure transform)')
    a4[2].plot(t, e2m, 'b', lw=1.0)
    a4[2].set_title(r'mean|e2| per step (→ e2_ss)')
    a4[3].plot(t, ee, 'b', lw=1.2, label='EE pos err')
    a4[3].plot(t, errp, 'g', lw=1.0, label='|e_p|')
    a4[3].plot(t, errr, 'r', lw=1.0, label='|e_R|')
    a4[3].set_title('EE errors (outer k_p loop still helps)')
    a4[3].legend(fontsize=8)
    a4[4].plot(t, jlm, 'b', lw=1.0)
    a4[4].axhline(LIM_MARGIN_REP, color='r', ls='--', lw=.8)
    a4[4].axhline(LIM_MARGIN_BRAKE, color='k', ls=':', lw=.8)
    a4[4].set_title('worst joint-limit margin (rad)')
    a4[5].plot(t, tau.max(axis=1), 'b', lw=.8, label='max|tau|')
    a4[5].plot(t, tau.min(axis=1), 'g', lw=.8, label='min|tau|')
    a4[5].set_title('torque envelope'); a4[5].legend(fontsize=8)
    for a in a4:
        a.grid(alpha=.3); a.set_xlabel('Time (s)')
    fig4.tight_layout()
    fig4.savefig('oneL_diag.png', dpi=300, bbox_inches='tight')
    print('[SAVED] oneL_diag.png')

    # --- fig5: 力矩 ---
    fig5, a5 = plt.subplots(2, 6, figsize=(16, 6))
    tl = [f'wheel {k}' for k in ['FR','FL','BR','BL']] + \
         [f'tau j{i+1}' for i in range(ctrl.num_arm)]
    caps = np.array([50.0]*4 + PANDA_TAU)
    for k in range(11):
        a5[k//6, k%6].plot(t, tau[:, k], 'b', lw=.8)
        a5[k//6, k%6].axhline( caps[k], color='r', ls='--', lw=.7)
        a5[k//6, k%6].axhline(-caps[k], color='r', ls='--', lw=.7)
        a5[k//6, k%6].set_title(tl[k]); a5[k//6, k%6].grid(alpha=.3)
    a5[0, 5].plot(t, errr, 'r', lw=1.0)
    a5[0, 5].set_title('$|e_R|$ (rad)'); a5[0, 5].grid(alpha=.3)
    a5[1, 5].axis('off')
    fig5.suptitle('Control Inputs (clips ON, matching [A+] baseline)',
                  fontsize=13)
    fig5.tight_layout()
    fig5.savefig('oneL_torques.png', dpi=300, bbox_inches='tight')
    print('[SAVED] oneL_torques.png')
    plt.show()

# ============================================================
# 8. 主循环
# ============================================================
def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    dt = model.opt.timestep

    cols = _jac_cols(model)
    scratch = mujoco.MjData(model)
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY_NAME)
    badr = model.jnt_dofadr[model.body_jntadr[b]]
    BASE_DOF_ADRS = [badr + 0, badr + 1, badr + 5]
    ARM_DOF_ADRS = [model.jnt_dofadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in ARM_JOINT_NAMES]
    ARM_JOINT_IDS = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                     for n in ARM_JOINT_NAMES]

    ctrl = MobileManipulatorPPC1LController(model, ARM_JOINT_NAMES)
    slew = SlewLimiter(ctrl.num_ctrl, S=SLEW_S) if SLEW_S else None

    traj_vis = [generate_trajectory(i*0.1)[0]
                for i in range(int(2*np.pi/0.1) + 1)]
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)

    rec = {k: [] for k in ['t','e1_diag','rho1_diag','e2','rho2','e2_mean',
                           'q','q_d_diag','tau','ee','w','errp','errr',
                           'mu2','jlim']}

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_forward(model, data)
        q_hold = np.array([data.qpos[a] for a in ctrl.arm_qpos_adrs])
        print(f'[INFO] Settling {T_SETTLE}s ...')
        for _ in range(int(T_SETTLE/dt)):
            mujoco.mj_forward(model, data)
            tau = ctrl.settle_step(data, q_hold)
            data.ctrl[:len(tau)] = tau
            mujoco.mj_step(model, data)

        mujoco.mj_forward(model, data)
        dq0 = null_space_ik_controller(
            model, data, scratch, cols, *generate_trajectory(0.0),
            BASE_DOF_ADRS, ARM_DOF_ADRS)[0]
        if slew: dq0 = slew(dq0, dt)
        ctrl.reset_states(data, dq_des=dq0)
        print(f'[1L] ONE-LAYER PPC (velocity only) | '
              f'INJECT_INTEGRAL={"ON (≈two-layer in disguise)" if INJECT_INTEGRAL else "OFF (pure one-layer)"} | '
              f'clips={"ON" if LIMIT_TORQUE else "OFF"} | '
              f'robust={"ON" if ROBUST_ON else "OFF"}')

        # 诊断参考积分器 (不进控制): q_d_diag(0) = q(0)
        q_d_diag = ctrl._get_q(data).copy()

        t, k = 0.0, 0
        while viewer.is_running() and t < T_SIM:
            mujoco.mj_forward(model, data)
            tp, tq, tv, two = generate_trajectory(t)

            dq_raw, w_now, lam_now, nspd, errp_n, errr_n, dqmax_n, diag = \
                null_space_ik_controller(
                    model, data, scratch, cols, tp, tq, tv, two,
                    BASE_DOF_ADRS, ARM_DOF_ADRS)

            dq = slew(dq_raw, dt) if slew else dq_raw
            ctrl.set_target(dq)
            tau = ctrl.compute_ctrl(data, dt)
            data.ctrl[:len(tau)] = tau

            # ---- 诊断积分 (纯记录): q_d_diag ← ∫ q̇_des ----
            q_d_diag[:2] += dt * ctrl.dq_des[:2]
            q_d_diag[2] = wrap_angle(q_d_diag[2] + dt*ctrl.dq_des[2])
            q_d_diag[3:] += dt * ctrl.dq_des[3:]

            if k % REC_SKIP == 0:
                qn = ctrl._get_q(data)
                e1d = qn - q_d_diag; e1d[2] = wrap_angle(e1d[2])
                jl, wj = joint_limit_margin(model, data,
                                            ctrl.arm_qpos_adrs,
                                            ARM_JOINT_IDS)
                e2n = ctrl.ppf2.domain_ok(
                    np.array([data.qvel[a] for a in ctrl.all_dof_adrs])
                    - ctrl.alpha_f, ctrl.t)
                e2v = (np.array([data.qvel[a] for a in ctrl.all_dof_adrs])
                       - ctrl.alpha_f)
                rec['t'].append(t)
                rec['e1_diag'].append(e1d.copy())
                rec['rho1_diag'].append(ctrl.ppf1_diag.f(ctrl.t).copy())
                rec['e2'].append(e2v.copy())
                rec['rho2'].append(ctrl.ppf2.f(ctrl.t).copy())
                rec['e2_mean'].append(float(np.mean(np.abs(e2v))))
                rec['q'].append(qn.copy())
                rec['q_d_diag'].append(q_d_diag.copy())
                rec['tau'].append(tau.copy())
                rec['ee'].append(np.linalg.norm(tp - data.xpos[ee_id]))
                rec['w'].append(w_now)
                rec['errp'].append(errp_n)
                rec['errr'].append(errr_n)
                rec['mu2'].append(float(np.max(np.abs(ctrl.mu2_last))))
                rec['jlim'].append(jl)

            viewer.user_scn.ngeom = 0
            for i in range(len(traj_vis)-1):
                add_line(viewer.user_scn, traj_vis[i], traj_vis[i+1],
                         0.005, np.array([1, 0, 0, 0.5]))
            if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.02, 0, 0]),
                    tp, np.eye(3).flatten(), np.array([1, 1, 0, 1]))
                viewer.user_scn.ngeom += 1

            mujoco.mj_step(model, data)
            viewer.sync()

            if k % PRT_SKIP == 0:
                print(f'  t={t:6.2f}s  ee_p={errp_n:.3f} m  '
                      f'ee_r={errr_n:.3f} rad  |dq|max={dqmax_n:.2f}  '
                      f'mean|e2|={rec["e2_mean"][-1]:.4f}  '
                      f'max|mu2|={rec["mu2"][-1]:.2f}  '
                      f'jlim={jl:.3f}rad(j{wj+1})  '
                      f'max|tau|={np.max(np.abs(tau)):.1f}')
            t += dt; k += 1

    # ================= [REPORT] =================
    print(f'\n[DONE] {len(rec["t"])} points.')
    print(f'[X3] L2 reinit events: {len(ctrl.reinit_events)}  |  '
          f'NaN events: {len(ctrl.nan_events)}')

    T = np.array(rec['t'])
    E1 = np.array(rec['e1_diag'])
    E2 = np.array(rec['e2'])
    R2 = 0.9*np.array(rec['rho2'])
    tail = T >= max(0.0, T[-1] - 20.0)
    names = ['base_x', 'base_y', 'yaw'] + \
            [f'j{i+1}' for i in range(ctrl.num_arm)]

    print('\n[REPORT] 速度漏斗 (一层方案唯一拥有的保证):')
    n_v2 = 0
    for i, nm in enumerate(names):
        v = float(np.max(np.abs(E2[tail, i]) - R2[tail, i]))
        if v > 0: n_v2 += 1
        print(f'   {nm:8s}  funnel margin={-v:+.4f} '
              f'({"OK" if v <= 0 else "VIOLATION"})')

    print('\n[REPORT] 位置漂移 (一层方案无法承诺的量) — 末20s线性拟合:')
    print('   channel   drift slope(/s)   mean|e2|(tail)   '
          'slope≈e2_ss?   total drift')
    for i, nm in enumerate(names):
        sl = float(np.polyfit(T[tail], E1[tail, i], 1)[0])
        e2ss = float(np.mean(np.abs(E2[tail, i])))
        tot = float(E1[-1, i] - E1[tail][0, i])
        flag = 'YES' if abs(sl) > 0.3*e2ss and e2ss > 1e-4 else \
               ('~0' if abs(sl) < 1e-4 else 'partial')
        print(f'   {nm:8s}   {sl:+.5f}          {e2ss:.5f}          '
              f'{flag:12s}   {tot:+.4f}')

    print('\n[REPORT] 关节限位余量: '
          f'min over run = {min(rec["jlim"]):.4f} rad '
          f'(若随时间下降 → 漂移正把关节推向限位)')

    if INJECT_INTEGRAL:
        print('\n[NOTE] INJECT_INTEGRAL=ON: ∫e₂ 已注入 α — '
              '若漂移斜率≈0, 即验证 "修好的一层 ≡ 两层变装" '
              '(∫e₂ 就是 e₁)')
    plot_results(rec, ctrl)

if __name__ == '__main__':
    main()
