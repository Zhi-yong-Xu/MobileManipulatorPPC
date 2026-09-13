#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[A++] 纯变换 PPC + 无输出级限幅 收敛实验
==========================================================================
保留: PPC 变换 μ=ln((0.9+φ)/(0.9−φ)) 全程纯解析, 无任何截断
      命令滤波反步结构, 鲁棒项 tanh(μ2/δ), 上游参考整形
      (Slew / v,ω 分通道饱和 / 限位刹车+排斥)
删除: [X2] 输出级所有限幅 np.clip (臂力矩/轮矩±50/伺服/settling)
      —— ctrllimited 执行器仍被 MuJoCo 物理钳制(删不掉), 其余真正无界
开关: [X3] REINIT_ON
      True  = 域外事件触发重初始化(q_d←q / α_f←q̇), 事件全记录  [A+机制]
      False = 裸 PPC: 越域→变换 NaN→零力矩(仅 NaN 守卫, 故障隔离)
              理论预言: 越域后 q_d 继续积分, e1 单调增大, 永久失控
诊断: 需求超旧上限率 / 重初始化或NaN事件流 / 漏斗违规(纯PPC应=0)
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
LIMIT_TORQUE = False    # [X2] False: 输出级不限幅
REINIT_ON    = True     # [X3] True: 重初始化机制; False: 裸PPC(越域即NaN失控)
ROBUST_ON    = True
WHEEL_SERVO  = False
SLEW_S       = 2.0

# ---- PPC 参数 ([A+] 纯变换版) ----
GAMMA1, GAMMA2 = 0.4, 3.0
EPSILON, DELTA = 0.15, 0.5
LAMBDA_F       = 20.0
KB0, KB1       = 2.0, 0.02

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
# 1b. [F5] 姿态误差 + 限位工具 (上游, 保留)
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
    for i, (adr, lim, lo, hi) in enumerate(_arm_limit_cache.get(id(model), [])):
        pass
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
# 2. PPF —— [A+] 纯变换, 无任何 clip
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
        """纯微分同胚, 无任何截断. 定义域外由调用方的前提监测负责."""
        ft = self.f(t)
        phi = e / ft
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.log((self.eps_up * (phi - self.eps_lo)) /
                          (-self.eps_lo * (self.eps_up - phi)))

def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi

# ============================================================
# 2b. Slew 限变化器 (上游参考整形, 保留: 定理假设P3)
# ============================================================
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
# 4. PPC 动力学层 ([A++] 纯变换 + 可关重初始化 + 无输出限幅)
# ============================================================
class MobileManipulatorPPCController:

    def __init__(self, model, arm_joint_names,
                 gamma1=GAMMA1, gamma2=GAMMA2, epsilon=EPSILON,
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

        self.gamma1, self.gamma2 = gamma1, gamma2
        self.epsilon, self.filter_gain = epsilon, filter_gain
        self.kb0, self.kb1 = kb0, kb1

        self.ctrl_lo, self.ctrl_hi = make_torque_limits(model, self.num_arm)

        f0_1  = np.array([0.20, 0.20, 0.30] + [0.25]*self.num_arm)
        fin_1 = np.array([0.05, 0.05, 0.05] + [0.025]*self.num_arm)
        self.ppf1 = PPFVector(f0_1, fin_1, 2.0, 10.0)
        f0_2  = np.array([2.5, 2.5, 2.0] + [3.0]*self.num_arm)
        fin_2 = np.array([0.10, 0.10, 0.10] + [0.1]*self.num_arm)
        self.ppf2 = PPFVector(f0_2, fin_2, 2.0, 10.0)

        self.dq_des = np.zeros(self.num_ctrl)
        self.q_d, self.alpha_f, self.alpha_f_dot = None, None, None
        self.t = 0.0
        self.mu1_last = np.zeros(self.num_ctrl)
        self.mu2_last = np.zeros(self.num_ctrl)

        # ---- 诊断 ----
        self.reinit_events = []                 # [X3] 跳转事件
        self.nan_events    = []                 # [X3] 裸PPC越域→NaN事件
        self.exceed_counts = np.zeros(self.num_out)   # [X2] 需求超旧上限
        self.tau_counts    = 0

    def set_target(self, dq_des):
        if len(dq_des) == self.num_ctrl:
            self.dq_des = np.array(dq_des)
        else:
            raise ValueError(f"期望维度 {self.num_ctrl}, 实际 {len(dq_des)}")

    def reset_states(self, data, dq_des=None):
        q = self._get_q(data)
        self.q_d = q.copy()              # (P1) e1(0)=0 精确
        if dq_des is not None:
            self.dq_des = np.array(dq_des)
        self.alpha_f = np.zeros(self.num_ctrl)
        self.alpha_f_dot = np.zeros(self.num_ctrl)
        self.t = 0.0
        self.reinit_events, self.nan_events = [], []
        self.exceed_counts[:] = 0
        self.tau_counts = 0

        dq0 = np.array([data.qvel[a] for a in self.all_dof_adrs])
        lim0 = 0.9 * self.ppf2.f(0.0)
        bad = np.where(np.abs(dq0) >= lim0)[0]
        if len(bad):
            print(f'[WARN] 前提P2违例 @handover: channels={bad.tolist()}')
        else:
            print(f'[PREMISE] P1: e1(0)=0 exact | '
                  f'P2: max|dq0|={np.max(np.abs(dq0)):.3f} < '
                  f'{np.min(lim0):.3f}  OK')

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
        return tau                                   # [X2] 无 clip

    def compute_ctrl(self, data, dt):
        q  = self._get_q(data)
        dq = np.array([data.qvel[a] for a in self.all_dof_adrs])
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(dq))):
            return np.zeros(self.num_out)            # NaN守卫: 故障隔离

        if self.q_d is None:
            self.q_d = q.copy()
            self.alpha_f = np.zeros(self.num_ctrl)
        self.q_d[:2] += dt * self.dq_des[:2]
        self.q_d[2] = wrap_angle(self.q_d[2] + dt*self.dq_des[2])
        self.q_d[3:] += dt * self.dq_des[3:]

        e1 = q - self.q_d
        e1[2] = wrap_angle(e1[2])

        # ---- [X3] Level-1 定义域处理 ----
        ok1 = self.ppf1.domain_ok(e1, self.t)
        if not np.all(ok1):
            v = np.where(~ok1)[0]
            phi = np.abs(e1) / self.ppf1.f(self.t)
            if REINIT_ON:
                # 跳转 (非限幅): 投影恢复前提(P1), 事件记录
                self.reinit_events.append(
                    (self.t, v.copy(), 1, float(phi[v].max())))
                print(f'[REINIT] t={self.t:.2f}s L1 ch={v.tolist()} '
                      f'max|phi|={phi[v].max():.3f} -> q_d<-q')
                self.q_d = q.copy()
                e1 = np.zeros_like(e1)
            # else: 裸PPC —— 不做任何处理, 变换将产生 NaN (下方捕获)

        mu1 = self.ppf1.transform(e1, self.t)
        self.mu1_last = mu1

        if np.all(np.isfinite(mu1)):
            alpha = self.dq_des - self.gamma1 * mu1
        else:
            if REINIT_ON:
                pass   # 理论上不应到达 (域内 log 恒有限)
            else:
                self.nan_events.append((self.t, np.where(~ok1)[0].copy(), 1))
                print(f'[NaN-FALLBACK] t={self.t:.2f}s L1 out-of-domain -> '
                      f'zero torque, control permanently lost '
                      f'(q_d keeps integrating, e1 grows: no recovery)')
            self.t += dt
            return np.zeros(self.num_out)   # 零力矩 = 无人驾驶

        self.alpha_f_dot = -self.filter_gain * (self.alpha_f - alpha)
        e2 = dq - self.alpha_f

        # ---- [X3] Level-2 定义域处理 ----
        ok2 = self.ppf2.domain_ok(e2, self.t)
        if not np.all(ok2):
            v = np.where(~ok2)[0]
            phi = np.abs(e2) / self.ppf2.f(self.t)
            if REINIT_ON:
                self.reinit_events.append(
                    (self.t, v.copy(), 2, float(phi[v].max())))
                print(f'[REINIT] t={self.t:.2f}s L2 ch={v.tolist()} '
                      f'max|phi|={phi[v].max():.3f} -> alpha_f<-dq')
                self.alpha_f = dq.copy()
                e2 = np.zeros_like(e2)

        mu2 = self.ppf2.transform(e2, self.t)
        self.mu2_last = mu2

        if not np.all(np.isfinite(mu2)):
            if not REINIT_ON:
                self.nan_events.append((self.t, np.where(~ok2)[0].copy(), 2))
                print(f'[NaN-FALLBACK] t={self.t:.2f}s L2 out-of-domain -> '
                      f'zero torque (permanent)')
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

        self.alpha_f += self.alpha_f_dot * dt
        self.t += dt

        tau = self._base_output(data, tau_sigma)
        if not np.all(np.isfinite(tau)):
            return np.zeros_like(tau)
        # ---- [X2] 输出级不限幅: 记录需求超旧上限 ----
        self.tau_counts += 1
        self.exceed_counts += ((tau > self.ctrl_hi + 1e-9) |
                               (tau < self.ctrl_lo - 1e-9))
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
            tau_w  = self.kp_w * (w_des - w_act)      # [X2] 无 clip
        else:
            tau_w = self.T_force @ (R_w2b @ tau_sigma[:3])   # [X2] 无 clip
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
    e1 = np.array(rec['e1']);  r1 = np.array(rec['rho1'])
    q  = np.array(rec['q']);   qd = np.array(rec['q_d'])
    tau = np.array(rec['tau'])
    w   = np.array(rec['w']);  lam = np.array(rec['lam'])
    ee  = np.array(rec['ee'])
    errp = np.array(rec['errp']); errr = np.array(rec['errr'])
    mu1m = np.array(rec['mu1']); mu2m = np.array(rec['mu2'])
    jlm = np.array(rec['jlim'])
    caps = np.array([50.0]*4 + PANDA_TAU)

    lbl = ['base x (m)', 'base y (m)', 'base yaw (rad)'] + \
          [f'panda j{i+1} (rad)' for i in range(ctrl.num_arm)]

    fig1, ax1 = plt.subplots(5, 2, figsize=(14, 12))
    for i in range(ctrl.num_ctrl):
        a = ax1[i//2, i%2]
        a.plot(t, e1[:, i], 'b', lw=1.0, label=fr'$e_1[{i+1}]$')
        a.plot(t,  0.9*r1[:, i], 'r--', lw=.8)
        a.plot(t, -0.9*r1[:, i], 'r--', lw=.8, label='funnel bound')
        a.fill_between(t, -0.9*r1[:, i], 0.9*r1[:, i], alpha=.08, color='r')
        a.set_title(lbl[i]); a.grid(alpha=.3); a.legend(fontsize=8)
    fig1.suptitle('Pure PPC (no clips anywhere in control law): '
                  'funnel guarantee should hold', fontsize=14)
    fig1.tight_layout()
    fig1.savefig('ppc_noclip_funnels.png', dpi=300, bbox_inches='tight')
    print('[SAVED] ppc_noclip_funnels.png')

    fig2, ax2 = plt.subplots(5, 2, figsize=(14, 12))
    for i in range(ctrl.num_ctrl):
        a = ax2[i//2, i%2]
        a.plot(t, qd[:, i], 'r--', lw=1.0, label='$q_d$')
        a.plot(t, q[:, i], 'b', lw=1.0, label='$q$')
        a.set_title(lbl[i]); a.grid(alpha=.3); a.legend(fontsize=8)
    fig2.suptitle('Reference Tracking (all 10 DOFs)', fontsize=14)
    fig2.tight_layout()
    fig2.savefig('ppc_noclip_tracking.png', dpi=300, bbox_inches='tight')
    print('[SAVED] ppc_noclip_tracking.png')

    fig3, a3 = plt.subplots(1, 6, figsize=(28, 3.5))
    a3[0].plot(t, w, 'b', lw=1.2)
    a3[0].axhline(0.02, color='r', ls='--', lw=.8)
    a3[0].set_title('Manipulability w(t)')
    a3[1].plot(t, lam, 'g', lw=1.2)
    a3[1].set_title(r'Adaptive damping $\lambda(t)$')
    a3[2].semilogy(t, mu1m, 'b', lw=1.0, label='max|mu1|')
    a3[2].semilogy(t, mu2m, 'r', lw=1.0, label='max|mu2|')
    a3[2].set_title('PPF transform (pure, unbounded)')
    a3[2].legend(fontsize=8)
    a3[3].plot(t, ee, 'b', lw=1.2)
    a3[3].set_title('EE pos error (m)')
    a3[4].plot(t, errp, 'b', lw=1.0, label='$|e_p|$')
    a3[4].plot(t, errr, 'r', lw=1.0, label='$|e_R|$')
    a3[4].set_title('EE errors'); a3[4].legend(fontsize=8)
    a3[5].plot(t, jlm, 'b', lw=1.0)
    a3[5].axhline(LIM_MARGIN_REP, color='r', ls='--', lw=.8)
    a3[5].axhline(LIM_MARGIN_BRAKE, color='k', ls=':', lw=.8)
    a3[5].set_title('worst joint-limit margin (rad)')
    for a in a3:
        a.grid(alpha=.3); a.set_xlabel('Time (s)')
    fig3.tight_layout()
    fig3.savefig('ppc_noclip_diag.png', dpi=300, bbox_inches='tight')
    print('[SAVED] ppc_noclip_diag.png')

    fig4, a4 = plt.subplots(2, 6, figsize=(16, 6))
    tl = [f'wheel {k}' for k in ['FR','FL','BR','BL']] + \
         [f'tau j{i+1}' for i in range(ctrl.num_arm)]
    for k in range(11):
        a4[k//6, k%6].plot(t, tau[:, k], 'b', lw=.8)
        a4[k//6, k%6].axhline( caps[k], color='r', ls='--', lw=.7,
                               label='old cap')
        a4[k//6, k%6].axhline(-caps[k], color='r', ls='--', lw=.7)
        a4[k//6, k%6].set_title(tl[k]); a4[k//6, k%6].grid(alpha=.3)
    a4[0, 5].plot(t, errr, 'r', lw=1.0)
    a4[0, 5].set_title('$|e_R|$ (rad)'); a4[0, 5].grid(alpha=.3)
    a4[1, 5].axis('off')
    fig4.suptitle('Control Inputs — red dashed = old clip levels '
                  '(demand beyond lines was previously absorbed by clip)',
                  fontsize=13)
    fig4.tight_layout()
    fig4.savefig('ppc_noclip_torques.png', dpi=300, bbox_inches='tight')
    print('[SAVED] ppc_noclip_torques.png')
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

    ctrl = MobileManipulatorPPCController(model, ARM_JOINT_NAMES)
    slew = SlewLimiter(ctrl.num_ctrl, S=SLEW_S) if SLEW_S else None

    n_lim = int(np.sum(model.actuator_ctrllimited[:ctrl.num_out]))
    print(f'[PHYS] ctrlrange-enforced clamp: {n_lim}/{ctrl.num_out} '
          f'actuators (物理饱和不可删除; 未limited的臂关节真正无界)')

    traj_vis = [generate_trajectory(i*0.1)[0]
                for i in range(int(2*np.pi/0.1) + 1)]
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)

    rec = {k: [] for k in ['t','e1','rho1','q','q_d','tau','ee','w','lam',
                           'errp','errr','mu1','mu2','jlim']}

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
        print(f'[A++] PURE PPC (transform: no clips) | '
              f'torque-clip={"ON" if LIMIT_TORQUE else "OFF"} | '
              f'REINIT={"ON" if REINIT_ON else "OFF (raw: NaN->zero torque)"}'
              f' | robust={"ON" if ROBUST_ON else "OFF"}')

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

            if k % REC_SKIP == 0:
                qn = ctrl._get_q(data)
                e1 = qn - ctrl.q_d; e1[2] = wrap_angle(e1[2])
                jl, wj = joint_limit_margin(model, data,
                                            ctrl.arm_qpos_adrs,
                                            ARM_JOINT_IDS)
                rec['t'].append(t);          rec['e1'].append(e1.copy())
                rec['rho1'].append(ctrl.ppf1.f(ctrl.t).copy())
                rec['q'].append(qn.copy());  rec['q_d'].append(ctrl.q_d.copy())
                rec['tau'].append(tau.copy())
                rec['ee'].append(np.linalg.norm(tp - data.xpos[ee_id]))
                rec['w'].append(w_now)
                rec['lam'].append(lam_now)
                rec['errp'].append(errp_n)
                rec['errr'].append(errr_n)
                rec['mu1'].append(float(np.max(np.abs(ctrl.mu1_last))))
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
                jam = int(np.argmax(np.abs(tau)))
                aname = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_ACTUATOR, jam) or f'#{jam}'
                print(f'  t={t:6.2f}s  ee_p={errp_n:.3f} m  '
                      f'ee_r={errr_n:.3f} rad  |dq|max={dqmax_n:.2f}  '
                      f'max|mu1|={rec["mu1"][-1]:.2f}  '
                      f'max|mu2|={rec["mu2"][-1]:.2f}  '
                      f'jlim={jl:.3f}rad(j{wj+1})  '
                      f'max|tau|@{aname}={np.max(np.abs(tau)):.1f}')
            t += dt; k += 1

    # ================= [REPORT] =================
    print(f'\n[DONE] {len(rec["t"])} points.')
    print(f'[X2] torque demand-over-old-cap rates '
          f'(输出级clip原本吸收的部分):')
    names_tau = ['wheelFR','wheelFL','wheelBR','wheelBL'] + \
                [f'j{i+1}' for i in range(ctrl.num_arm)]
    for i in range(ctrl.num_out):
        if ctrl.tau_counts and ctrl.exceed_counts[i] > 0:
            print(f'   {names_tau[i]:8s}  '
                  f'{ctrl.exceed_counts[i]/ctrl.tau_counts:.4f}  '
                  f'({int(ctrl.exceed_counts[i])}/{ctrl.tau_counts} steps, '
                  f'cap=±{ctrl.ctrl_hi[i]:.0f})')
    if np.all(ctrl.exceed_counts == 0):
        print('   none — 需求全程未超旧上限')

    if REINIT_ON:
        print(f'\n[X3] re-initialization events: {len(ctrl.reinit_events)}')
        if not ctrl.reinit_events:
            print('   none — 漏斗保证全程未被打破: '
                  '纯PPC在名义工况下无需任何机制即收敛')
        else:
            for (te, chs, lv, ph) in ctrl.reinit_events[:20]:
                print(f'   t={te:6.2f}s L{lv} ch={chs.tolist()} '
                      f'max|phi|={ph:.3f}')
    else:
        print(f'\n[X3] NaN-fallback events: {len(ctrl.nan_events)}')
        if not ctrl.nan_events:
            print('   none — 裸PPC全程未越域, 与REINIT_ON=True轨迹完全一致')
        else:
            print('   越域发生后变换永久NaN -> 零力矩 -> 失控 '
                  '(这就是最小机制存在的证明)')

    if rec['t']:
        E1 = np.array(rec['e1']); R1 = 0.9*np.array(rec['rho1'])
        t_arr = np.array(rec['t'])
        over = np.maximum(E1 - R1, -E1 - R1)
        tail = t_arr >= max(0.0, t_arr[-1] - 5.0)
        names = ['base_x', 'base_y', 'yaw'] + \
                [f'j{i+1}' for i in range(ctrl.num_arm)]
        print('[REPORT] funnel violations (纯PPC应全为0 = 屏障机制独立生效):')
        n_viol = 0
        for i, nm in enumerate(names):
            v = float(over[:, i].max())
            ss = float(np.mean(np.abs(E1[tail, i])))
            flag = ''
            if v > 0:
                n_viol += 1
                flag = '   <-- VIOLATION'
            print(f'   {nm:8s}  over={v:+.4f}  |e1|ss={ss:.4f}{flag}')
        if n_viol == 0:
            print('   全程在漏斗内 — 无限幅PPC收敛成立, 屏障即保证')
        plot_results(rec, ctrl)

if __name__ == '__main__':
    main()
