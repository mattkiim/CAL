"""CAL-compatible F16 stabilize-avoid environment.

This module adapts the RAC F16 stabilize-avoid v6 task to the lightweight CAL
environment contract used by ``Quad2DEnv`` and ``Quad3DEnv``:

    obs, reward, cost, terminated, truncated, info = env.step(action)

The dynamics path uses ``jax_f16.f16.F16`` when the environment is constructed.
Install ``jax`` and ``jax_f16`` before training or evaluating this env.
"""

from __future__ import annotations

from typing import Dict, Optional

import gym
import gym.spaces
import numpy as np


IDX_VT, IDX_ALPHA, IDX_BETA = 0, 1, 2
IDX_PHI, IDX_THETA, IDX_PSI = 3, 4, 5
IDX_P, IDX_Q, IDX_R = 6, 7, 8
IDX_PN, IDX_PE, IDX_H = 9, 10, 11
IDX_POW = 12
STATE_DIM = 16

INIT_STATE_KEYS = [
    "init_vt",
    "init_alpha",
    "init_beta",
    "init_phi",
    "init_theta",
    "init_psi",
    "init_p",
    "init_q",
    "init_r",
    "init_pn",
    "init_pe",
    "init_h",
    "init_pow",
]
INIT_KEY_TO_IDX = {
    "init_vt": IDX_VT,
    "init_alpha": IDX_ALPHA,
    "init_beta": IDX_BETA,
    "init_phi": IDX_PHI,
    "init_theta": IDX_THETA,
    "init_psi": IDX_PSI,
    "init_p": IDX_P,
    "init_q": IDX_Q,
    "init_r": IDX_R,
    "init_pn": IDX_PN,
    "init_pe": IDX_PE,
    "init_h": IDX_H,
    "init_pow": IDX_POW,
}

CONTROL_LOW = np.array([-10.0, -10.0, -10.0, 0.0], dtype=np.float32)
CONTROL_HIGH = np.array([15.0, 10.0, 10.0, 1.0], dtype=np.float32)

ANGLE_IDXS = np.array([IDX_PHI, IDX_THETA, IDX_PSI], dtype=np.int64)
OTHER_IDXS = np.array(
    [
        IDX_VT,
        IDX_ALPHA,
        IDX_BETA,
        IDX_P,
        IDX_Q,
        IDX_R,
        IDX_PN,
        IDX_PE,
        IDX_H,
        IDX_POW,
        13,
        14,
        15,
    ],
    dtype=np.int64,
)


def _require_f16_model():
    try:
        from jax_f16.f16 import F16
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "F16StabilizeEnv requires the optional dependency 'jax_f16'. "
            "Install jax and jax_f16 before using env_name=F16."
        ) from exc
    return F16()


def nominal_state_v5() -> np.ndarray:
    """Nominal straight-and-level F16 state used for resets/eval starts."""

    x = np.zeros(STATE_DIM, dtype=np.float64)
    x[IDX_VT] = 540.0
    x[IDX_ALPHA] = 0.0
    x[IDX_BETA] = 0.0
    x[IDX_PHI] = 0.0
    x[IDX_THETA] = 0.0
    x[IDX_PSI] = 0.0
    x[IDX_P] = 0.0
    x[IDX_Q] = 0.0
    x[IDX_R] = 0.0
    x[IDX_PN] = 0.0
    x[IDX_PE] = 0.0
    x[IDX_H] = 500.0
    x[IDX_POW] = 9.0
    return x


def state_to_options(state: np.ndarray) -> Dict[str, float]:
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    if state.shape[0] < STATE_DIM:
        raise ValueError(f"Expected F16 state with at least {STATE_DIM} dims, got {state.shape}")
    return {key: float(state[idx]) for key, idx in INIT_KEY_TO_IDX.items()}


def options_to_state(options: Optional[Dict[str, float]]) -> np.ndarray:
    state = nominal_state_v5()
    if options:
        for key, idx in INIT_KEY_TO_IDX.items():
            if key in options:
                state[idx] = float(options[key])
    return state


class F16StabilizeEnv:
    """F16 stabilize-avoid task with CAL-compatible observations and costs."""

    def __init__(
        self,
        dt: float = 0.05,
        max_episode_steps: int = 640,
        seed: int = 0,
        goal_dwell_steps: int = 50,
        normalize_obs: bool = True,
        obs_task_feats_mode: str = "per_axis",
        reset_box_mode: str = "ours",
        init_curriculum: bool = True,
        init_curriculum_mode: str = "box",
        init_curriculum_frac: float = 0.5,
        init_curriculum_frac_end: float = 0.5,
        init_curriculum_anneal_start: int = 200_000,
        init_curriculum_anneal_end: int = 800_000,
        region_profile: str = "v6",
        safety_gap_mode: str = "raw",
        l_form: str = "split_log",
        l_q_center: float = 1.0,
        l_q_out: float = 20.0,
        l_kappa: float = 2.0,
        l_stage_reward: float = 0.0,
        action_penalty: float = 1e-3,
        crash_reward_penalty: float = 500.0,
    ):
        del region_profile, init_curriculum_mode
        self.dt = float(dt)
        self.max_episode_steps = int(max_episode_steps)
        self.goal_dwell_steps = int(goal_dwell_steps)
        self.obs_task_feats_mode = obs_task_feats_mode
        self.reset_box_mode = reset_box_mode
        self.init_curriculum = bool(init_curriculum)
        self.init_curriculum_frac = float(init_curriculum_frac)
        self.init_curriculum_frac_end = float(init_curriculum_frac_end)
        self.init_curriculum_anneal_start = int(init_curriculum_anneal_start)
        self.init_curriculum_anneal_end = int(init_curriculum_anneal_end)
        self.safety_gap_mode = safety_gap_mode
        self.l_form = l_form
        self.l_q_center = float(l_q_center)
        self.l_q_out = float(l_q_out)
        self.l_kappa = float(l_kappa)
        self.l_stage_reward = float(l_stage_reward)
        self.action_penalty = float(action_penalty)
        self.crash_reward_penalty = float(crash_reward_penalty)

        self.goal_h_mid = 100.0
        self.goal_h_halfwidth = 25.0
        self.goal_h_min = self.goal_h_mid - self.goal_h_halfwidth
        self.goal_h_max = self.goal_h_mid + self.goal_h_halfwidth

        self.safe_h_min = 50.0
        self.safe_h_max = 1000.0
        self.safe_alpha_lo = -0.5
        self.safe_alpha_hi = 0.5
        self.safe_beta = 0.5
        self.safe_theta = 1.1
        self.safe_pe = 1000.0
        self.safe_p = 2.0

        self.terminate_h_min = 0.0
        self.terminate_h_max = 1100.0
        self.terminate_alpha_lo = -0.7
        self.terminate_alpha_hi = 0.7
        self.terminate_beta = 0.7
        self.terminate_theta = 1.4
        self.terminate_pe_limit = 1200.0
        self.terminate_p = 3.0
        self.h_theta_denom = self.terminate_theta - self.safe_theta

        self.alpha_counts_as_invalid_dynamics = True
        self.beta_counts_as_invalid_dynamics = True

        obs_dim = 22 + (8 if obs_task_feats_mode == "per_axis" else 2)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-np.ones(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )

        self._f16 = _require_f16_model()
        self._rng = np.random.default_rng(seed)
        self._x = nominal_state_v5()
        self._t = 0
        self._train_step = 0
        self._goal_streak = 0
        self._max_goal_streak = 0

        self._episode_reward = 0.0
        self._episode_cost = 0.0
        self._episode_length = 0

        self._normalize_obs = bool(normalize_obs)
        self._obs_norm_clip = 10.0
        self._obs_mean = np.zeros(obs_dim, dtype=np.float64)
        self._obs_var = np.ones(obs_dim, dtype=np.float64)
        self._obs_count = 0
        self._update_stats = True

    @property
    def state(self) -> np.ndarray:
        return self._x.astype(np.float32).copy()

    def nominal_state_v5(self) -> np.ndarray:
        return nominal_state_v5()

    def set_train_step(self, step: int) -> None:
        self._train_step = int(step)

    def set_eval_mode(self) -> None:
        self._update_stats = False

    def set_train_mode(self) -> None:
        self._update_stats = True

    def get_obs_stats(self):
        return self._obs_mean.copy(), self._obs_var.copy(), int(self._obs_count)

    def set_obs_stats(self, mean, var, count) -> None:
        self._obs_mean = np.asarray(mean, dtype=np.float64).copy()
        self._obs_var = np.asarray(var, dtype=np.float64).copy()
        self._obs_count = int(count)

    def _decode_action(self, action: np.ndarray):
        action_clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        control = 0.5 * (CONTROL_HIGH - CONTROL_LOW) * action_clipped
        control = control + 0.5 * (CONTROL_HIGH + CONTROL_LOW)
        return action_clipped, control.astype(np.float32)

    def _rk4(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        import jax
        import jax.numpy as jnp

        x = jnp.asarray(state, dtype=jnp.float32)
        u = jnp.asarray(control, dtype=jnp.float32)
        k1 = self._f16.xdot(x, u)
        k2 = self._f16.xdot(x + 0.5 * self.dt * k1, u)
        k3 = self._f16.xdot(x + 0.5 * self.dt * k2, u)
        k4 = self._f16.xdot(x + self.dt * k3, u)
        x_next = x + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return np.asarray(jax.device_get(x_next), dtype=np.float64)

    def _task_goal_distance(self, x: np.ndarray) -> float:
        return float(max(abs(float(x[IDX_H]) - self.goal_h_mid) - self.goal_h_halfwidth, 0.0) / 250.0)

    def _task_goal_bool(self, x: np.ndarray) -> bool:
        return self._task_goal_distance(x) <= 0.0

    def _task_l_value(self, x: np.ndarray) -> float:
        z = (float(x[IDX_H]) - self.goal_h_mid) / self.goal_h_halfwidth
        d_out = max(abs(z) - 1.0, 0.0)
        l_center = self.l_q_center * min(z * z, 1.0)
        if self.l_form == "split_linear":
            phi = d_out
        elif self.l_form == "split_atan":
            phi = np.arctan(self.l_kappa * d_out) / self.l_kappa
        else:
            phi = np.log1p(self.l_kappa * d_out) / self.l_kappa
        l_base = l_center + self.l_q_out * phi
        if self.l_stage_reward > 0.0:
            l_base -= self.l_stage_reward * max(1.0 - z * z, 0.0)
        return float(l_base)

    def _task_h_components_raw(self, x: np.ndarray) -> np.ndarray:
        h_altitude = max(
            (self.safe_h_min - x[IDX_H]) / (self.safe_h_min - self.terminate_h_min),
            (x[IDX_H] - self.safe_h_max) / (self.terminate_h_max - self.safe_h_max),
        )
        h_alpha = max(
            (self.safe_alpha_lo - x[IDX_ALPHA]) / (self.safe_alpha_lo - self.terminate_alpha_lo),
            (x[IDX_ALPHA] - self.safe_alpha_hi) / (self.terminate_alpha_hi - self.safe_alpha_hi),
        )
        h_beta = (abs(x[IDX_BETA]) - self.safe_beta) / (self.terminate_beta - self.safe_beta)
        h_theta = (abs(x[IDX_THETA]) - self.safe_theta) / self.h_theta_denom
        h_pe = (abs(x[IDX_PE]) - self.safe_pe) / (self.terminate_pe_limit - self.safe_pe)
        h_p = (abs(x[IDX_P]) - self.safe_p) / (self.terminate_p - self.safe_p)
        return np.asarray([h_altitude, h_alpha, h_beta, h_theta, h_pe, h_p], dtype=np.float64)

    def _task_h_components(self, x: np.ndarray) -> np.ndarray:
        h_raw = self._task_h_components_raw(x)
        if self.safety_gap_mode == "literal":
            return np.where(h_raw >= 0.0, h_raw + 0.5, h_raw - 0.5)
        return h_raw

    def _task_safety_margin(self, x: np.ndarray) -> float:
        return float(np.max(self._task_h_components(x)))

    def _is_valid(self, x: np.ndarray) -> bool:
        finite_ok = bool(np.all(np.isfinite(x)) and np.isfinite(x[IDX_H]) and np.isfinite(x[IDX_PE]))
        alpha_ok = self.terminate_alpha_lo <= x[IDX_ALPHA] <= self.terminate_alpha_hi
        beta_ok = abs(x[IDX_BETA]) <= self.terminate_beta
        theta_ok = abs(x[IDX_THETA]) < self.terminate_theta
        p_ok = abs(x[IDX_P]) < self.terminate_p
        if not self.alpha_counts_as_invalid_dynamics:
            alpha_ok = True
        if not self.beta_counts_as_invalid_dynamics:
            beta_ok = True
        return bool(finite_ok and alpha_ok and beta_ok and theta_ok and p_ok)

    def _hits_hard_terminal(self, x: np.ndarray) -> bool:
        return bool(
            x[IDX_H] <= self.terminate_h_min
            or x[IDX_H] >= self.terminate_h_max
            or abs(x[IDX_PE]) >= self.terminate_pe_limit
        )

    def _classify_crash_cause(self, x: np.ndarray) -> str:
        checks = [
            ("altitude_low", x[IDX_H] <= self.terminate_h_min),
            ("altitude_high", x[IDX_H] >= self.terminate_h_max),
            ("pe", abs(x[IDX_PE]) >= self.terminate_pe_limit),
            ("alpha", not (self.terminate_alpha_lo <= x[IDX_ALPHA] <= self.terminate_alpha_hi)),
            ("beta", abs(x[IDX_BETA]) > self.terminate_beta),
            ("theta", abs(x[IDX_THETA]) >= self.terminate_theta),
            ("p", abs(x[IDX_P]) >= self.terminate_p),
        ]
        for name, hit in checks:
            if hit:
                return name
        return "none"

    def simulate_transition(self, state: np.ndarray, action: np.ndarray) -> Dict[str, object]:
        action_clipped, control = self._decode_action(action)
        x_raw = self._rk4(np.asarray(state, dtype=np.float64), control)
        h_components = self._task_h_components(x_raw)
        h_margin = float(np.max(h_components))
        task_cost = self._task_l_value(x_raw)
        action_cost = self.action_penalty * float(np.sum(np.square(action_clipped)))
        reward = -(task_cost + action_cost)
        goal_distance = self._task_goal_distance(x_raw)
        in_goal = goal_distance <= 0.0
        safe = h_margin <= 0.0
        invalid_dynamics = not self._is_valid(x_raw)
        hard_terminal = self._hits_hard_terminal(x_raw)
        terminated = bool(invalid_dynamics or hard_terminal)
        return {
            "next_state": x_raw,
            "raw_next_state": x_raw,
            "action_clipped": action_clipped,
            "control": control,
            "h_components": h_components,
            "h_margin": h_margin,
            "task_cost": float(task_cost),
            "reward": float(np.clip(reward, -100.0, 10.0)),
            "binary_cost": float(h_margin > 0.0),
            "goal_distance": float(goal_distance),
            "in_goal": bool(in_goal),
            "safe": bool(safe),
            "invalid_dynamics": bool(invalid_dynamics),
            "hard_terminal": bool(hard_terminal),
            "terminated": bool(terminated),
            "crash_cause": self._classify_crash_cause(x_raw) if terminated else "none",
        }

    def _compute_vel_angles(self, x: np.ndarray) -> np.ndarray:
        vt = max(abs(float(x[IDX_VT])), 1e-6)
        return np.asarray(
            [
                np.sin(float(x[IDX_ALPHA])),
                np.sin(float(x[IDX_BETA])),
                float(x[IDX_H]) / 1000.0,
            ],
            dtype=np.float32,
        ) * np.asarray([1.0, 1.0, min(vt / 540.0, 2.0)], dtype=np.float32)

    def _state_enc(self, x: np.ndarray) -> np.ndarray:
        state = np.asarray(x, dtype=np.float64)
        angles = state[ANGLE_IDXS]
        other = state[OTHER_IDXS]
        enc = np.concatenate([other, np.cos(angles), np.sin(angles), self._compute_vel_angles(state)])
        scale = np.asarray(
            [
                540.0,
                0.5,
                0.5,
                2.0,
                2.0,
                2.0,
                1000.0,
                1000.0,
                1000.0,
                10.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float64,
        )
        return np.clip(enc / scale, -10.0, 10.0).astype(np.float32)

    def _update_obs_running_stats(self, obs: np.ndarray) -> None:
        if not self._update_stats:
            return
        self._obs_count += 1
        delta = obs.astype(np.float64) - self._obs_mean
        self._obs_mean += delta / self._obs_count
        delta2 = obs.astype(np.float64) - self._obs_mean
        self._obs_var += (delta * delta2 - self._obs_var) / self._obs_count

    def _norm_obs(self, obs: np.ndarray) -> np.ndarray:
        if not self._normalize_obs:
            return obs.astype(np.float32)
        self._update_obs_running_stats(obs)
        std = np.sqrt(self._obs_var + 1e-8)
        normed = (obs.astype(np.float64) - self._obs_mean) / std
        return np.clip(normed, -self._obs_norm_clip, self._obs_norm_clip).astype(np.float32)

    def _get_obs(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        state = self._x if x is None else np.asarray(x, dtype=np.float64)
        h_components = self._task_h_components(state)
        h_margin = float(np.max(h_components))
        task_feats = [np.tanh(h_margin), np.tanh(self._task_goal_distance(state))]
        if self.obs_task_feats_mode == "per_axis":
            task_feats.extend(np.tanh(h_components).tolist())
        obs = np.concatenate([self._state_enc(state), np.asarray(task_feats, dtype=np.float32)])
        return self._norm_obs(obs.astype(np.float32))

    def _curriculum_frac(self) -> float:
        if not self.init_curriculum:
            return self.init_curriculum_frac_end
        if self.init_curriculum_anneal_end <= self.init_curriculum_anneal_start:
            return self.init_curriculum_frac_end
        p = (self._train_step - self.init_curriculum_anneal_start) / (
            self.init_curriculum_anneal_end - self.init_curriculum_anneal_start
        )
        p = float(np.clip(p, 0.0, 1.0))
        return (1.0 - p) * self.init_curriculum_frac + p * self.init_curriculum_frac_end

    def _sample_reset_state(self) -> np.ndarray:
        base = nominal_state_v5()
        frac = self._curriculum_frac()
        theta_width = 0.15 + 0.65 * frac
        h_low = 500.0 - 450.0 * frac
        h_high = 500.0 + 500.0 * frac
        for _ in range(2000):
            x = base.copy()
            x[IDX_ALPHA] = self._rng.uniform(-0.1, 0.1)
            x[IDX_BETA] = self._rng.uniform(-0.05, 0.05)
            x[IDX_THETA] = self._rng.uniform(-theta_width, theta_width)
            x[IDX_P] = self._rng.uniform(-0.2, 0.2)
            x[IDX_Q] = self._rng.uniform(-0.2, 0.2)
            x[IDX_R] = self._rng.uniform(-0.2, 0.2)
            x[IDX_PE] = self._rng.uniform(-250.0 * frac, 250.0 * frac)
            x[IDX_H] = self._rng.uniform(h_low, h_high)
            if self._task_safety_margin(x) < 0.0:
                return x
        raise RuntimeError("Failed to sample safe F16 initial state.")

    def sample_x0_eval_diag_v5(self, num: int, seed: int = 0, safe_margin_max: float = -0.2) -> np.ndarray:
        rng = np.random.default_rng(seed)
        theta_vals = np.linspace(-0.6, 0.6, 21)
        h_vals = np.linspace(100.0, 900.0, 21)
        states = []
        base = nominal_state_v5()
        for theta in theta_vals:
            for h_alt in h_vals:
                x = base.copy()
                x[IDX_THETA] = float(theta)
                x[IDX_H] = float(h_alt)
                if self._task_safety_margin(x) < safe_margin_max:
                    states.append(x)
        rng.shuffle(states)
        return np.asarray(states[: int(num)], dtype=np.float64)

    def diagnostic_grid_v5(self):
        theta_vals = np.linspace(-0.6, 0.6, 21)
        h_vals = np.linspace(100.0, 900.0, 21)
        grid = np.empty((len(theta_vals), len(h_vals), STATE_DIM), dtype=np.float64)
        base = nominal_state_v5()
        for i, theta in enumerate(theta_vals):
            for j, h_alt in enumerate(h_vals):
                x = base.copy()
                x[IDX_THETA] = float(theta)
                x[IDX_H] = float(h_alt)
                grid[i, j] = x
        return theta_vals, h_vals, grid

    def reset(self, seed=None, options: Optional[Dict[str, float]] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._x = options_to_state(options) if options is not None else self._sample_reset_state()
        self._t = 0
        self._goal_streak = 0
        self._max_goal_streak = 0
        self._episode_reward = 0.0
        self._episode_cost = 0.0
        self._episode_length = 0
        return self._get_obs(), {}

    def step(self, action):
        trans = self.simulate_transition(self._x, action)
        self._x = np.asarray(trans["next_state"], dtype=np.float64)
        h_val = float(trans["h_margin"])
        cost = float(trans["binary_cost"])
        reward = float(trans["reward"])
        in_goal = bool(trans["in_goal"])

        if in_goal and h_val <= 0.0:
            self._goal_streak += 1
        else:
            self._goal_streak = 0
        self._max_goal_streak = max(self._max_goal_streak, self._goal_streak)

        success = self._goal_streak >= self.goal_dwell_steps
        terminated = bool(trans["terminated"] or success)
        self._t += 1
        truncated = bool(self._t >= self.max_episode_steps and not terminated)

        crashed = bool(trans["terminated"] and not success)
        if crashed:
            remaining_frac = (self.max_episode_steps - self._t) / float(self.max_episode_steps)
            reward -= self.crash_reward_penalty * (1.0 + max(remaining_frac, 0.0))

        self._episode_reward += reward
        self._episode_cost += cost
        self._episode_length += 1

        info = {
            "h": h_val,
            "cost": cost,
            "cost_dense": h_val,
            "binary_cost": cost,
            "raw_state": self._x.astype(np.float32).copy(),
            "control": np.asarray(trans["control"], dtype=np.float32),
            "h_components": np.asarray(trans["h_components"], dtype=np.float32),
            "task_cost": float(trans["task_cost"]),
            "goal_distance": float(trans["goal_distance"]),
            "in_goal": in_goal,
            "success": bool(success),
            "crashed": crashed,
            "crash_cause": str(trans["crash_cause"]),
            "invalid_dynamics": bool(trans["invalid_dynamics"]),
            "hard_terminal": bool(trans["hard_terminal"]),
            "max_goal_streak": int(self._max_goal_streak),
            "terminated": terminated,
            "truncated": truncated,
            "episode_reward": float(self._episode_reward),
            "episode_cost": float(self._episode_cost),
            "episode_length": int(self._episode_length),
        }
        return self._get_obs(), reward, cost, terminated, truncated, info

    def close(self) -> None:
        return None

    @property
    def episode_info(self):
        return {
            "reward": self._episode_reward,
            "cost": self._episode_cost,
            "length": self._episode_length,
        }


def perturbation_grid_init_states(num_episodes: int) -> np.ndarray:
    perturbations = [
        (0.0, 500.0),
        (0.4, 500.0),
        (-0.4, 500.0),
        (0.0, 200.0),
        (0.0, 900.0),
        (0.6, 400.0),
        (-0.6, 600.0),
        (0.0, 100.0),
    ]
    states = []
    base = nominal_state_v5()
    for i in range(int(num_episodes)):
        delta_theta, h_alt = perturbations[i % len(perturbations)]
        x = base.copy()
        x[IDX_THETA] = float(base[IDX_THETA] + delta_theta)
        x[IDX_H] = float(h_alt)
        states.append(x)
    return np.asarray(states, dtype=np.float64)
