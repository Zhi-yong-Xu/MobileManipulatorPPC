#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
移动机械臂 PPC/DSC + 零空间奇异规避 仿真 — SummitXL + Panda
========================================================
运动学层:
    轨迹 -> 分通道误差 -> (v_ff + Kp_p*e_p | w_ff + Kp_r*e_r)
    [F5] 姿态误差世界系计算(不依赖 mju_subQuat 约定), 双覆盖最短路径
    分通道限幅 [FIX-2]: |v|<=0.6, |w|<=1.0 (映射前)
    主任务:  自适应阻尼 DLS
    零空间:  可操作度梯度上升(e_p门控) + [B] 限位排斥(不门控)
             + [B2] 限位刹车(禁止朝壁速度指令)
    dq_des 经 Slew 限变化率
动力学层: USE_PPC 切换 PPC / DSC
  [S1] gamma1=0.4, rho1_inf 加宽 (G1 <= filter/4 裕度准则)
  [A]  去掉 mu 外层 clip(±1): 恢复 atanh 微分同胚与边界排斥
       (内层数值保险保留: mu 事实封顶 ±7.5, 边界最大修正 gamma1*7.5)
  [S3] ssign delta=0.5, epsilon=0.15
底盘层:   [FIX-1] 力分配 T_force  或  [FIX-3] 轮速伺服; 输出 4轮+7臂=11 维
安全层:   力矩按 ctrlrange 硬限幅 / Kb 封顶 / PD settling
诊断:     dFRM / w∥ / jlim / [C] 终局漏斗违规报告(逐通道+稳态偏移)
========================================================
"""
import os
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

USE_PPC     = True
WHEEL_SERVO = False
SLEW_S      = 2.0

# [B] 限位回避参数
LIM_MARGIN_REP   = 0.15   # 排斥区半宽 (rad)
K_LIM            = 2.0    # 排斥峰值速度 (rad/s)
LIM_MARGIN_BRAKE = 0.05   # 刹车区半宽 (rad)

# ============================================================
# 1. 轨迹
# ============================================================
CENTER = np.array([0.0, 0.0, 0.7])
RADIUS, OMEGA = 1.0, 0.2
TARGET_QUAT = np.array([0, 1, 0, 0])     # 手爪z轴朝下

def generate_trajectory(t):
    th = OMEGA * t
    pos = np.array([CENTER[0] + RADIUS*np.cos(th),
                    CENTER[1] + RADIUS*np.sin(th), CENTER[2]])
    vel = np.array([-RADIUS*OMEGA*np.sin(th),
                     RADIUS*OMEGA*np.cos(th), 0.0])
    return pos, TARGET_QUAT.copy(), vel, np.zeros(3)

# ============================================================
# 1b. [F5] 世界系姿态误差 + 限位工具
# ============================================================
def orientation_error_world(tq, qc):
    """世界系旋转向量: qe = tq ⊗ qc*, 双覆盖取最短路径."""
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
    """[(qpos_adr, limited, lo, hi)] x7, 按 model 缓存"""
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
    """[B1] 臂关节限位排斥速度(关节空间): 距任一限位<dm 时平方律内推.
       返回 (7维速度, 激活关节计数)"""
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
    """[B2] 刹车: 距限位<dm 时禁止朝壁方向的dq分量 (dq长度10, 臂在[3:10])"""
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
# 2. PPF  ([A] 无外层 clip)
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

    def transform(self, e, t):
        ft = self.f(t)
        # 内层数值保险: 防 |e|>=0.9rho 时 atanh 发散; mu 事实封顶 +-7.5
        phi = np.clip(e / ft, self.eps_lo + 1e-3, self.eps_up - 1e-3)
        mu = np.log((self.eps_up * (phi - self.eps_lo)) /
                    (-self.eps_lo * (self.eps_up - phi)))
        return mu                    # [A] 无外层 clip: 真微分同胚+边界排斥

def ssign(x, delta=0.5):                       # [S3]
    return np.tanh(x / delta)

def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi

# ============================================================
# 2b. Slew 限变化器
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
# 4. PPC 动力学层
# ============================================================
class MobileManipulatorPPCController:

    def __init__(self, model, arm_joint_names,
                 gamma1=0.4, gamma2=3.0, epsilon=0.15,
                 filter_gain=20.0, kb0=2.0, kb1=0.02):
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
        self.sat_hist = []

        self.gamma1, self.gamma2 = gamma1, gamma2
        self.epsilon, self.filter_gain = epsilon, filter_gain
        self.kb0, self.kb1 = kb0, kb1

        f0_1  = np.array([0.20, 0.20, 0.30] + [0.25]*self.num_arm)
        fin_1 = np.array([0.15, 0.15, 0.20] + [0.12]*self.num_arm)
        self.ppf1 = PPFVector(f0_1, fin_1, 2.0, 10.0)
        f0_2  = np.array([2.5, 2.5, 2.0] + [3.0]*self.num_arm)
        fin_2 = np.array([0.20, 0.20, 0.20] + [0.25]*self.num_arm)
        self.ppf2 = PPFVector(f0_2, fin_2, 2.0, 10.0)

        self.ctrl_lo, self.ctrl_hi = make_torque_limits(model, self.num_arm)

        self.dq_des = np.zeros(self.num_ctrl)
        self.q_d, self.alpha_f, self.alpha_f_dot = None, None, None
        self.t = 0.0
        self.mu1_last = np.zeros(self.num_ctrl)
        self.mu2_last = np.zeros(self.num_ctrl)

    def set_target(self, dq_des):
        if len(dq_des) == self.num_ctrl:
            self.dq_des = np.array(dq_des)
        else:
            raise ValueError(f"期望维度 {self.num_ctrl}, 实际 {len(dq_des)}")

    def reset_states(self, data, dq_des=None):
        q = self._get_q(data)
        self.q_d = q.copy()
        if dq_des is not None:
            self.dq_des = np.array(dq_des)
        self.alpha_f = np.zeros(self.num_ctrl)
        self.alpha_f_dot = np.zeros(self.num_ctrl)
        self.t = 0.0

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
        return np.clip(tau, self.ctrl_lo, self.ctrl_hi)

    def compute_ctrl(self, data, dt):
        q  = self._get_q(data)
        dq = np.array([data.qvel[a] for a in self.all_dof_adrs])
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(dq))):
            return np.zeros(self.num_out)

        if self.q_d is None:
            self.q_d = q.copy()
            self.alpha_f = np.zeros(self.num_ctrl)
        self.q_d[:2] += dt * self.dq_des[:2]
        self.q_d[2] = wrap_angle(self.q_d[2] + dt*self.dq_des[2])
        self.q_d[3:] += dt * self.dq_des[3:]

        e1 = q - self.q_d
        e1[2] = wrap_angle(e1[2])
        mu1 = self.ppf1.transform(e1, self.t)
        self.mu1_last = mu1
        alpha = self.dq_des - self.gamma1 * mu1

        self.alpha_f_dot = -self.filter_gain * (self.alpha_f - alpha)

        e2 = dq - self.alpha_f
        mu2 = self.ppf2.transform(e2, self.t)
        self.mu2_last = mu2

        mujoco.mj_fullM(self.model, data, self.M_full)
        idx = np.array(self.all_dof_adrs)
        M = self.M_full[np.ix_(idx, idx)]
        h = np.array([data.qfrc_bias[a] - data.qfrc_passive[a]
                      for a in self.all_dof_adrs])

        vnorm = min(np.linalg.norm(dq), 100.0)
        Kb = self.kb0 + self.kb1 * vnorm ** 2

        s2s = ssign(mu2)
        tau_sigma = h + M @ (self.alpha_f_dot - self.gamma2*mu2
                             - self.epsilon*s2s) - Kb * s2s

        self.alpha_f += self.alpha_f_dot * dt
        self.t += dt

        tau = self._base_output(data, tau_sigma)
        if not np.all(np.isfinite(tau)):
            return np.zeros_like(tau)
        return np.clip(tau, self.ctrl_lo, self.ctrl_hi)

    def _base_output(self, data, tau_sigma):
        xm = data.xmat[self.base_body_id].reshape(3, 3)
        yaw = np.arctan2(xm[1, 0], xm[0, 0])
        R_w2b = np.array([[ np.cos(yaw), np.sin(yaw), 0],
                          [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        if WHEEL_SERVO:
            u_body = R_w2b @ self.dq_des[:3]
            w_des  = self.J_wheels @ u_body
            w_act  = np.array([data.qvel[a] for a in self.wheel_dof_adrs])
            tau_w  = np.clip(self.kp_w * (w_des - w_act), -50.0, 50.0)
        else:
            tau_w = np.clip(self.T_force @ (R_w2b @ tau_sigma[:3]),
                            -50.0, 50.0)
            self.sat_hist.append(float(np.mean(np.abs(tau_w) >= 49.9)))
        return np.concatenate((tau_w, tau_sigma[3:]))


# ============================================================
# 4b. DSC 动力学层 (A/B 基线, 不改动)
# ============================================================
class MobileManipulatorDSCController:

    def __init__(self, model, arm_joint_names,
                 filter_gain=20.0, kd=5.0, safe_clamp=True):
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

        self.filter_gain = filter_gain
        self.kd = kd
        self.safe_clamp = safe_clamp

        self.alpha   = np.zeros(self.num_ctrl)
        self.dq_prev = np.zeros(self.num_ctrl)
        self.dq_des  = np.zeros(self.num_ctrl)

        self.q_d = None
        self.t = 0.0
        self.mu1_last = np.zeros(self.num_ctrl)
        self.mu2_last = np.zeros(self.num_ctrl)

        f0  = np.array([0.30, 0.30, 0.50] + [0.50]*self.num_arm)
        fin = np.array([0.05, 0.05, 0.08] + [0.08]*self.num_arm)
        self.ppf1 = PPFVector(f0, fin, 2.0, 10.0)

        self.J_wheels = J_WHEELS
        self.T_force  = T_FORCE
        self.wheel_dof_adrs = [model.jnt_dofadr[model.actuator_trnid[i, 0]]
                               for i in range(4)]
        self.kp_w = KP_W
        self.sat_hist = []

        self.ctrl_lo, self.ctrl_hi = make_torque_limits(model, self.num_arm)

    def set_target(self, dq_des):
        if len(dq_des) == self.num_ctrl:
            self.dq_des = np.array(dq_des)
        else:
            raise ValueError(f"期望维度 {self.num_ctrl}, 实际 {len(dq_des)}")

    def reset_states(self, data, dq_des=None):
        q = self._get_q(data)
        self.q_d = q.copy()
        if dq_des is not None:
            self.dq_des = np.array(dq_des)
        self.alpha   = self.dq_des.copy()
        self.dq_prev = self.dq_des.copy()
        self.t = 0.0

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
        return np.clip(tau, self.ctrl_lo, self.ctrl_hi)

    def compute_ctrl(self, data, dt):
        dq_actual = np.array([data.qvel[a] for a in self.all_dof_adrs])
        d_alpha = -self.filter_gain * (self.alpha - self.dq_des)
        ddq = (self.dq_des - self.dq_prev) / dt   # noqa: F841

        mujoco.mj_fullM(self.model, data, self.M_full)
        idx = np.array(self.all_dof_adrs)
        M_ctrl = self.M_full[np.ix_(idx, idx)]

        tau_bias    = np.array([data.qfrc_bias[a]    for a in self.all_dof_adrs])
        tau_passive = np.array([data.qfrc_passive[a] for a in self.all_dof_adrs])

        tau_virtual = (tau_bias - tau_passive) + M_ctrl @ (
            d_alpha + self.kd * (self.dq_des - dq_actual))

        self.alpha += d_alpha * dt
        self.dq_prev = self.dq_des.copy()

        if self.q_d is None:
            self.q_d = self._get_q(data)
        self.q_d[:2] += dt * self.dq_des[:2]
        self.q_d[2] = wrap_angle(self.q_d[2] + dt * self.dq_des[2])
        self.q_d[3:] += dt * self.dq_des[3:]
        self.t += dt

        tau = self._base_output(data, tau_virtual)
        if not np.all(np.isfinite(tau)):
            return np.zeros_like(tau)
        if self.safe_clamp:
            tau = np.clip(tau, self.ctrl_lo, self.ctrl_hi)
        return tau

    def _base_output(self, data, tau_virtual):
        xm = data.xmat[self.base_body_id].reshape(3, 3)
        yaw = np.arctan2(xm[1, 0], xm[0, 0])
        R_w2b = np.array([[ np.cos(yaw), np.sin(yaw), 0],
                          [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        if WHEEL_SERVO:
            u_body = R_w2b @ self.dq_des[:3]
            w_des  = self.J_wheels @ u_body
            w_act  = np.array([data.qvel[a] for a in self.wheel_dof_adrs])
            tau_w  = np.clip(self.kp_w * (w_des - w_act), -50.0, 50.0)
        else:
            tau_w = np.clip(self.T_force @ (R_w2b @ tau_virtual[:3]),
                            -50.0, 50.0)
            self.sat_hist.append(float(np.mean(np.abs(tau_w) >= 49.9)))
        return np.concatenate((tau_w, tau_virtual[3:]))

# ============================================================
# 5. 雅可比 + 奇异性度量 + 零空间规避
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

# ------------------------------------------------------------
# 6. 外层 IK ([F5] + [B1]/[B2] 限位回避 + 诊断)
# ------------------------------------------------------------
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
    err_subq = np.zeros(3)
    mujoco.mju_subQuat(err_subq, tq, data.xquat[ee].copy())
    frame_diff = float(np.linalg.norm(err_subq - err_r))

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

    # ---- 零空间投影算子 (无条件下, 供梯度项与限位排斥共用) ----
    N = np.eye(3 + 7) - np.linalg.pinv(J, rcond=1e-6) @ J

    dq_null = np.zeros(3 + 7)

    # ---- 可操作度梯度 (e_p 门控) ----
    k_eff = k_null * np.clip((0.5 - float(np.linalg.norm(err_p))) / 0.3,
                             0.0, 1.0)
    if k_eff > 1e-6 and w > 0.3 * w_thr:
        _, g = manipulability_gradient(model, data, scratch, cols,
                                       base_dof_adrs, arm_dof_adrs,
                                       with_rotation=with_rotation)
        gn = np.linalg.norm(g)
        if gn > 1e-12:
            dq_null += k_eff * (N @ (g / gn))

    # ---- [B1] 限位排斥 (不受 e_p 门控, 安全优先) ----
    v_rep, n_rep = null_space_limit_repulsion(model, data)
    if n_rep > 0:
        rep10 = np.zeros(3 + 7)
        rep10[3:] = v_rep
        dq_null += N @ rep10

    dq = dq_task + dq_null

    # ---- [B2] 限位刹车: 禁止朝壁速度指令 ----
    dq = apply_limit_braking(model, data, dq)

    # ---- 诊断 ----
    dq_now = np.array([data.qvel[c] for c in cols])
    omega_real = J[3:] @ dq_now
    ern = float(np.linalg.norm(err_r))
    w_along = float(np.dot(omega_real, err_r / ern)) if ern > 0.05 else 0.0

    diag = {'frame_diff': frame_diff, 'w_along': w_along, 'n_rep': n_rep}
    return (dq, w, np.sqrt(lam2), float(np.linalg.norm(dq_null)),
            float(np.linalg.norm(err_p)), ern,
            float(np.max(np.abs(dq))), diag)

# ============================================================
# 7. 可视化辅助
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
# 8. 绘图
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
    nsp = np.array(rec['nullspd'])
    ee  = np.array(rec['ee'])
    errp = np.array(rec['errp']); errr = np.array(rec['errr'])
    mu1m = np.array(rec['mu1']); mu2m = np.array(rec['mu2'])
    fd  = np.array(rec['frame_diff']); wal = np.array(rec['w_along'])
    jlm = np.array(rec['jlim'])

    lbl = ['base x (m)', 'base y (m)', 'base yaw (rad)'] + \
          [f'panda j{i+1} (rad)' for i in range(ctrl.num_arm)]

    fig1, ax1 = plt.subplots(5, 2, figsize=(14, 12))
    for i in range(ctrl.num_ctrl):
        a = ax1[i//2, i%2]
        a.plot(t, e1[:, i], 'b', lw=1.0, label=fr'$e_1[{i+1}]$')
        a.plot(t,  0.9*r1[:, i], 'r--', lw=.8)
        a.plot(t, -0.9*r1[:, i], 'r--', lw=.8, label='PPF bound')
        a.fill_between(t, -0.9*r1[:, i], 0.9*r1[:, i], alpha=.08, color='r')
        a.set_title(lbl[i]); a.grid(alpha=.3); a.legend(fontsize=8)
    fig1.suptitle('Level-1 Errors inside Prescribed Performance Funnels',
                  fontsize=14)
    fig1.tight_layout()
    fig1.savefig('ppc_funnels.png', dpi=300, bbox_inches='tight')
    print('[SAVED] ppc_funnels.png')

    fig2, ax2 = plt.subplots(5, 2, figsize=(14, 12))
    for i in range(ctrl.num_ctrl):
        a = ax2[i//2, i%2]
        a.plot(t, qd[:, i], 'r--', lw=1.0, label='$q_d$')
        a.plot(t, q[:, i], 'b', lw=1.0, label='$q$')
        a.set_title(lbl[i]); a.grid(alpha=.3); a.legend(fontsize=8)
    fig2.suptitle('Reference Tracking (all 10 DOFs)', fontsize=14)
    fig2.tight_layout()
    fig2.savefig('ppc_tracking.png', dpi=300, bbox_inches='tight')
    print('[SAVED] ppc_tracking.png')

    fig3, a3 = plt.subplots(1, 6, figsize=(28, 3.5))
    a3[0].plot(t, w, 'b', lw=1.2)
    a3[0].axhline(0.02, color='r', ls='--', lw=.8, label='$w_{thr}$')
    a3[0].set_title('Manipulability  w(t)'); a3[0].legend(fontsize=8)
    a3[1].plot(t, lam, 'g', lw=1.2)
    a3[1].set_title(r'Adaptive damping $\lambda(t)$')
    a3[2].plot(t, nsp, 'm', lw=1.0)
    a3[2].set_title(r'null-space speed (gated)')
    a3[3].plot(t, mu1m, 'b', lw=1.0, label='max|mu1|')
    a3[3].plot(t, mu2m, 'r', lw=1.0, label='max|mu2|')
    a3[3].axhline(7.5, color='k', ls=':', lw=.8, label='guard cap ~7.5')
    a3[3].set_title('PPF transform (unclipped)'); a3[3].legend(fontsize=8)
    a3[4].plot(t, wal, 'b', lw=1.0, label=r'$\omega_\parallel$ (along err)')
    a3[4].plot(t, fd, 'r', lw=1.0, label='frame diff')
    a3[4].set_title('[F5] rotation diagnostics'); a3[4].legend(fontsize=8)
    a3[5].plot(t, jlm, 'b', lw=1.0)
    a3[5].axhline(LIM_MARGIN_REP, color='r', ls='--', lw=.8, label='rep zone')
    a3[5].axhline(LIM_MARGIN_BRAKE, color='k', ls=':', lw=.8, label='brake')
    a3[5].set_title('worst joint-limit margin (rad)'); a3[5].legend(fontsize=8)
    for a in a3:
        a.grid(alpha=.3); a.set_xlabel('Time (s)')
    fig3.tight_layout()
    fig3.savefig('ppc_singularity.png', dpi=300, bbox_inches='tight')
    print('[SAVED] ppc_singularity.png')

    fig4, a4 = plt.subplots(2, 6, figsize=(16, 6))
    tl = [f'wheel {k}' for k in ['FR','FL','BR','BL']] + \
         [f'tau j{i+1}' for i in range(ctrl.num_arm)]
    for k in range(11):
        a4[k//6, k%6].plot(t, tau[:, k], 'b', lw=.8)
        a4[k//6, k%6].set_title(tl[k]); a4[k//6, k%6].grid(alpha=.3)
    a4[0, 5].plot(t, ee, 'b', lw=1.2)
    a4[0, 5].set_title('EE pos error (m)'); a4[0, 5].grid(alpha=.3)
    a4[1, 5].plot(t, errp, 'b', lw=1.0, label='$|e_p|$ (m)')
    a4[1, 5].plot(t, errr, 'r', lw=1.0, label='$|e_R|$ (rad)')
    a4[1, 5].set_title('EE pos/orient error'); a4[1, 5].grid(alpha=.3)
    a4[1, 5].legend(fontsize=8)
    fig4.suptitle('Control Inputs & End-Effector Errors', fontsize=14)
    fig4.tight_layout()
    fig4.savefig('ppc_torques.png', dpi=300, bbox_inches='tight')
    print('[SAVED] ppc_torques.png')
    plt.show()

# ============================================================
# 9. 主循环
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

    ctrl = (MobileManipulatorPPCController(model, ARM_JOINT_NAMES)
            if USE_PPC else
            MobileManipulatorDSCController(model, ARM_JOINT_NAMES))

    slew = SlewLimiter(ctrl.num_ctrl, S=SLEW_S) if SLEW_S else None

    traj_vis = [generate_trajectory(i*0.1)[0]
                for i in range(int(2*np.pi/0.1) + 1)]
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)

    rec = {k: [] for k in ['t','e1','rho1','q','q_d','tau','zeta',
                           'zeta_d','ee','w','lam','nullspd',
                           'errp','errr','dqmax','mu1','mu2',
                           'frame_diff','w_along','jlim']}

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_forward(model, data)
        q_hold = np.array([data.qpos[a] for a in ctrl.arm_qpos_adrs])
        print(f'[INFO] Settling {T_SETTLE}s (PD + gravity comp) ...')
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
        print(f'[INFO] Handover to {"PPC" if USE_PPC else "DSC"} '
              f'| wheels={"servo" if WHEEL_SERVO else "T_force"} '
              f'| quat={TARGET_QUAT} | mu UNCLIPPED [A] '
              f'| lim-rep dm={LIM_MARGIN_REP} k={K_LIM} [B]')

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

            if k == 0:
                print(f'[CHECK] tau.shape={tau.shape} '
                      f'ctrl_lo.shape={ctrl.ctrl_lo.shape} '
                      f'model.nu={model.nu}')
                assert tau.shape == ctrl.ctrl_lo.shape

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
                rec['zeta'].append(np.array(
                    [data.qvel[a] for a in ctrl.base_dof_adrs]))
                rec['zeta_d'].append(np.array([tv[0], tv[1], two[2]]))
                rec['ee'].append(np.linalg.norm(tp - data.xpos[ee_id]))
                rec['w'].append(w_now)
                rec['lam'].append(lam_now)
                rec['nullspd'].append(nspd)
                rec['errp'].append(errp_n)
                rec['errr'].append(errr_n)
                rec['dqmax'].append(dqmax_n)
                rec['mu1'].append(float(np.max(np.abs(ctrl.mu1_last))))
                rec['mu2'].append(float(np.max(np.abs(ctrl.mu2_last))))
                rec['frame_diff'].append(diag['frame_diff'])
                rec['w_along'].append(diag['w_along'])
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
                      f'w={w_now:.3f}  w∥={diag["w_along"]:+.2f}  '
                      f'dFRM={diag["frame_diff"]:.3f}  '
                      f'nrep={diag["n_rep"]}  '
                      f'jlim={jl:.3f}rad(j{wj+1})  '
                      f'max|tau|@{aname}={np.max(np.abs(tau)):.1f}')
            t += dt; k += 1

    print(f'\n[DONE] {len(rec["t"])} points.')
    if ctrl.sat_hist:
        print(f'[DIAG] wheel torque saturation rate = '
              f'{np.mean(ctrl.sat_hist):.3f}')

    # ---- [C] 终局漏斗违规报告: 逐通道 + 稳态偏移 ----
    if rec['t']:
        E1 = np.array(rec['e1']); R1 = 0.9*np.array(rec['rho1'])
        t_arr = np.array(rec['t'])
        over = np.maximum(E1 - R1, -E1 - R1)      # >0 即出带
        tail = t_arr >= max(0.0, t_arr[-1] - 5.0)
        names = ['base_x', 'base_y', 'yaw'] + \
                [f'j{i+1}' for i in range(ctrl.num_arm)]
        print('[REPORT] funnel violations per channel '
              '(max over bound | steady-state |e1|, last 5s):')
        for i, nm in enumerate(names):
            v = float(over[:, i].max())
            ss = float(np.mean(np.abs(E1[tail, i])))
            flag = '   <-- VIOLATION' if v > 0 else ''
            print(f'   {nm:8s}  over={v:+.4f}  |e1|ss={ss:.4f}{flag}')
        plot_results(rec, ctrl)

if __name__ == '__main__':
    main()
