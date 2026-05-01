import numpy as np
import gym
import gym.spaces

GRAV = 9.80665


def quad3d_xdot(state, u, m=1.0):
    px, py, pz, vx, vy, vz, phi, theta, psi = state
    thrust, phi_dot, theta_dot, psi_dot = u

    s_theta, c_theta = np.sin(theta), np.cos(theta)
    s_phi, c_phi = np.sin(phi), np.cos(phi)

    xdot = np.zeros(9, dtype=np.float64)
    xdot[0] = vx
    xdot[1] = vy
    xdot[2] = vz
    xdot[3] = -(thrust / m) * s_theta
    xdot[4] = (thrust / m) * c_theta * s_phi
    xdot[5] = GRAV - (thrust / m) * c_theta * c_phi
    xdot[6] = phi_dot
    xdot[7] = theta_dot
    xdot[8] = psi_dot
    return xdot


def quad3d_step(state, u, dt, m=1.0):
    return (state + dt * quad3d_xdot(state, u, m)).astype(np.float32)


class Quad3DEnv:
    PX, PY, PZ = 0, 1, 2
    VX, VY, VZ = 3, 4, 5
    PHI, THETA, PSI = 6, 7, 8

    def __init__(
        self,
        dt=0.01,
        max_episode_steps=500,
        seed=0,
        action_penalty=1e-3,
        q_pos=10.0,
        q_vel=1.0,
        q_phi=1.0,
        q_theta=0.02,
        q_psi=0.02,
        target_pz=0.0,
        crash_reward_penalty=1000.0,
        normalize_obs=False,
    ):
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.m = 1.0

        self.thrust_scale = self.m * GRAV
        self.rate_scale = 5.0
        self.u_eq = np.array([self.m * GRAV, 0.0, 0.0, 0.0], dtype=np.float32)

        self.x_ref = np.zeros(9, dtype=np.float32)
        self.x_ref[self.PZ] = float(target_pz)

        self.Q_diag = np.array(
            [q_pos, q_pos, q_pos, q_vel, q_vel, q_vel, q_phi, q_theta, q_psi],
            dtype=np.float32,
        )
        self.Q = np.diag(self.Q_diag).astype(np.float32)
        self.R = np.diag([action_penalty] * 4).astype(np.float32)
        self.crash_reward_penalty = float(crash_reward_penalty)

        self.safe_pz = 0.0
        self.safe_radius = 3.0
        self.unsafe_pz = 0.3
        self.unsafe_radius = 3.5

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-np.ones(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self.state = np.zeros(9, dtype=np.float32)
        self._t = 0

        self._episode_reward = 0.0
        self._episode_cost = 0.0
        self._episode_length = 0

        self._normalize_obs = normalize_obs
        self._obs_norm_clip = 5.0
        self._obs_mean = np.zeros(9, dtype=np.float64)
        self._obs_var = np.ones(9, dtype=np.float64)
        self._obs_count = 0
        self._update_stats = True

        self._reset_low = np.array(
            [-1.5, -1.5, -1.0, -0.5, -0.5, -0.5, -0.2, -0.2, -0.2],
            dtype=np.float32,
        )
        self._reset_high = np.array(
            [1.5, 1.5, -0.05, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2],
            dtype=np.float32,
        )

    def _remap_action(self, action):
        a = np.clip(action, -1.0, 1.0)
        u = np.zeros(4, dtype=np.float64)
        u[0] = self.thrust_scale * (1.0 + a[0])
        u[1:] = self.rate_scale * a[1:]
        return u

    def _sdf(self, state):
        pz = float(state[self.PZ])
        state_norm = float(np.linalg.norm(state))
        return max(pz - self.safe_pz, state_norm - self.safe_radius)

    def _is_unsafe(self, state):
        pz = float(state[self.PZ])
        state_norm = float(np.linalg.norm(state))
        return (pz >= self.unsafe_pz) or (state_norm >= self.unsafe_radius)

    def _is_safe_start(self, state):
        return self._sdf(state) < 0.0

    def _update_obs_running_stats(self, obs):
        if not self._update_stats:
            return
        self._obs_count += 1
        delta = obs.astype(np.float64) - self._obs_mean
        self._obs_mean += delta / self._obs_count
        delta2 = obs.astype(np.float64) - self._obs_mean
        self._obs_var += (delta * delta2 - self._obs_var) / self._obs_count

    def _norm_obs(self, obs):
        if not self._normalize_obs:
            return obs.astype(np.float32)
        self._update_obs_running_stats(obs)
        std = np.sqrt(self._obs_var + 1e-8)
        obs = (obs.astype(np.float64) - self._obs_mean) / std
        return np.clip(obs, -self._obs_norm_clip, self._obs_norm_clip).astype(np.float32)

    def _get_obs(self):
        return self._norm_obs(self.state.copy())

    def set_eval_mode(self):
        self._update_stats = False

    def set_train_mode(self):
        self._update_stats = True

    def get_obs_stats(self):
        return self._obs_mean.copy(), self._obs_var.copy(), self._obs_count

    def set_obs_stats(self, mean, var, count):
        self._obs_mean = mean.copy()
        self._obs_var = var.copy()
        self._obs_count = int(count)

    def _compute_reward(self, state, action_norm):
        # state_err = state.astype(np.float32) - self.x_ref
        # action_norm = action_norm.astype(np.float32)
        # state_cost = float(state_err @ self.Q @ state_err)
        # action_cost = float(action_norm @ self.R @ action_norm)
        # reward = -(state_cost + action_cost)
        
        state_err = state.astype(np.float32) - self.x_ref
        action_norm = action_norm.astype(np.float32)

        state_cost = float(np.sum(self.Q_diag * np.abs(state_err)))
        action_cost = float(1e-3 * np.sum(np.abs(action_norm)))

        reward = -(state_cost + action_cost)
                
        return float(np.clip(reward, -100.0, 0.0))

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if options is not None:
            self.state = np.array(
                [
                    options.get("init_px", 0.0),
                    options.get("init_py", 0.0),
                    options.get("init_pz", -0.5),
                    options.get("init_vx", 0.0),
                    options.get("init_vy", 0.0),
                    options.get("init_vz", 0.0),
                    options.get("init_phi", 0.0),
                    options.get("init_theta", 0.0),
                    options.get("init_psi", 0.0),
                ],
                dtype=np.float32,
            )
        else:
            for _ in range(1000):
                candidate = self._rng.uniform(self._reset_low, self._reset_high).astype(np.float32)
                if self._is_safe_start(candidate):
                    self.state = candidate
                    break
            else:
                raise RuntimeError("Failed to sample safe Quad3D initial state.")

        self._t = 0
        self._episode_reward = 0.0
        self._episode_cost = 0.0
        self._episode_length = 0

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        u_physical = self._remap_action(action)
        self.state = quad3d_step(self.state, u_physical, self.dt, self.m)

        h_val = self._sdf(self.state)
        binary_cost = float(h_val > 0.0)
        # binary_cost = max(0.0, float(h_val))

        reward = self._compute_reward(self.state, action)

        terminated = bool(self._is_unsafe(self.state))
        self._t += 1
        truncated = bool(self._t >= self.max_episode_steps)

        if terminated:
            remaining_frac = (self.max_episode_steps - self._t) / float(self.max_episode_steps)
            reward -= self.crash_reward_penalty * (1.0 + max(remaining_frac, 0.0))

        # if terminated:
        #     binary_cost += 10.0
        # binary_cost = float(np.clip(binary_cost, 0.0, 20.0))

        self._episode_reward += reward
        self._episode_cost += binary_cost
        self._episode_length += 1

        info = {
            "h": float(h_val),
            "cost": float(binary_cost),
            "cost_dense": float(h_val),
            "binary_cost": float(binary_cost),
            "pz": float(self.state[self.PZ]),
            "state_norm": float(np.linalg.norm(self.state)),
            "terminated": terminated,
            "truncated": truncated,
            "episode_reward": float(self._episode_reward),
            "episode_cost": float(self._episode_cost),
            "episode_length": int(self._episode_length),
        }

        obs = self._get_obs()
        return obs, reward, binary_cost, terminated, truncated, info

    @property
    def episode_info(self):
        return {
            "reward": self._episode_reward,
            "cost": self._episode_cost,
            "length": self._episode_length,
        }
