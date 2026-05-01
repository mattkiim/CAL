#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from arguments import readParser
from agent.cal import CALAgent
from env.quad3d import Quad3DEnv


"""
Example:

python eval/quad3d_eval.py \
  --model_dir models/Quad3D_exp1 \
  --suffix 8272_epoch100 \
  --out_dir eval_cal_quad3d \
  --n_eval 100
"""


def resolve_model_paths(model_dir: str, suffix: str | None):
    if suffix is not None:
        return (
            os.path.join(model_dir, f"actor_{suffix}"),
            os.path.join(model_dir, f"critics_{suffix}"),
            os.path.join(model_dir, f"safetycritics_{suffix}"),
            suffix,
        )

    candidates = [f for f in os.listdir(model_dir) if f.startswith("actor_")]
    if not candidates:
        raise FileNotFoundError(f"No actor_* files found in {model_dir}")

    latest = max(candidates, key=lambda f: os.path.getmtime(os.path.join(model_dir, f)))
    suffix = latest[len("actor_"):]
    print(f"Auto-selected suffix: {suffix!r}")

    return (
        os.path.join(model_dir, f"actor_{suffix}"),
        os.path.join(model_dir, f"critics_{suffix}"),
        os.path.join(model_dir, f"safetycritics_{suffix}"),
        suffix,
    )


def make_cal_args(cost_lim: float, epoch_length: int):
    old_argv = sys.argv
    sys.argv = [sys.argv[0]]
    args = readParser()
    sys.argv = old_argv

    args.env_name = "Quad3D"
    args.safetygym = True
    args.constraint_type = "safetygym"
    args.cost_lim = cost_lim
    args.epoch_length = epoch_length
    return args


def sample_valid_init(rng, x_range, y_range, z_range, max_tries=10000):
    for _ in range(max_tries):
        init = {
            "init_px": float(rng.uniform(*x_range)),
            "init_py": float(rng.uniform(*y_range)),
            "init_pz": float(rng.uniform(*z_range)),
            "init_vx": 0.0,
            "init_vy": 0.0,
            "init_vz": 0.0,
            "init_phi": 0.0,
            "init_theta": 0.0,
            "init_psi": 0.0,
        }

        s = np.array(
            [
                init["init_px"],
                init["init_py"],
                init["init_pz"],
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

        # Quad3D convention: pz > 0 is downward / unsafe.
        h = max(float(s[2]), float(np.linalg.norm(s) - 3.0))
        unsafe = (s[2] >= 0.3) or (np.linalg.norm(s) >= 3.5)

        if h < 0.0 and not unsafe:
            return init

    raise RuntimeError("Failed to sample valid Quad3D init.")


def state_to_init_options(state: np.ndarray) -> Dict[str, float]:
    return {
        "init_px": float(state[0]),
        "init_py": float(state[1]),
        "init_pz": float(state[2]),
        "init_vx": float(state[3]),
        "init_vy": float(state[4]),
        "init_vz": float(state[5]),
        "init_phi": float(state[6]),
        "init_theta": float(state[7]),
        "init_psi": float(state[8]),
    }


def load_init_states(path: str, key: str, n_eval: int | None):
    data = np.load(path)
    if key not in data:
        raise KeyError(f"{path} missing key={key!r}. Available: {list(data.keys())}")

    states = np.asarray(data[key], dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 9:
        raise ValueError(f"Expected {key!r} to have shape (N, 9), got {states.shape}")

    if n_eval is not None:
        states = states[:n_eval]

    return [state_to_init_options(state) for state in states]


def rollout_one(env: Quad3DEnv, agent: CALAgent, init: Dict[str, float], seed: int):
    obs, _ = env.reset(seed=seed, options=init)

    states, next_states, actions, rewards, costs, hs = [], [], [], [], [], []

    terminated = False
    truncated = False

    while not (terminated or truncated):
        raw_state = env.state.copy()
        action = np.asarray(agent.select_action(obs, eval=True), dtype=np.float32)

        states.append(raw_state)
        actions.append(action)

        out = env.step(action)
        if len(out) != 6:
            raise ValueError(f"Expected 6 returns from Quad3DEnv.step, got {len(out)}")

        obs, reward, cost, terminated, truncated, info = out

        next_states.append(env.state.copy())
        rewards.append(float(reward))
        costs.append(float(cost))
        hs.append(float(info.get("h", np.nan)))

    states = np.asarray(states, dtype=np.float32)
    next_states = np.asarray(next_states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)
    costs = np.asarray(costs, dtype=np.float32)
    h_arr = np.asarray(hs, dtype=np.float32)

    length = int(len(rewards))

    # True violation definition:
    # violated state iff h(x_t) > 0
    unsafe_mask = h_arr > 0.0
    unsafe_steps = int(np.sum(unsafe_mask))
    violated_trajectory = bool(unsafe_steps > 0)

    terminal_state = next_states[-1] if len(next_states) else np.zeros(9, dtype=np.float32)

    terminal_state_norm = float(np.linalg.norm(terminal_state))
    terminal_pos_norm = float(np.linalg.norm(terminal_state[:3]))
    terminal_vel_norm = float(np.linalg.norm(terminal_state[3:6]))
    terminal_ang_norm = float(np.linalg.norm(terminal_state[6:9]))
    terminal_pz = float(terminal_state[2])
    terminal_h = float(max(terminal_pz, terminal_state_norm - 3.0))
    ground_crash = bool(terminated and terminal_pz >= env.unsafe_pz)
    radius_crash = bool(terminated and terminal_state_norm >= env.unsafe_radius)

    return {
        "init": init,

        "states": states,
        "next_states": next_states,
        "actions": actions,
        "rewards": rewards,
        "costs": costs,
        "h": h_arr,

        "return": float(rewards.sum()),
        "cost_sum": float(costs.sum()),

        # These are diagnostics for cost. Cost may include crash penalty.
        "cost_rate": float(costs.mean()) if len(costs) else float("nan"),

        # These are the requested safety metrics at rollout level.
        "unsafe_steps": unsafe_steps,
        "violated_trajectory": violated_trajectory,

        "length": length,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "crashed": bool(terminated),

        "terminal_state_norm": terminal_state_norm,
        "terminal_pos_norm": terminal_pos_norm,
        "terminal_vel_norm": terminal_vel_norm,
        "terminal_ang_norm": terminal_ang_norm,
        "terminal_pz": terminal_pz,
        "terminal_h": terminal_h,
        "ground_crash": ground_crash,
        "radius_crash": radius_crash,
    }


def tail_window(values: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(values)
    return values[-min(int(n), len(values)) :]


def quad3d_ref(env: Quad3DEnv) -> np.ndarray:
    return np.asarray(getattr(env, "x_ref", np.zeros(9, dtype=np.float32)), dtype=np.float32)


def terminal_state_error(rollout: Dict, x_ref: np.ndarray) -> float:
    states = np.asarray(rollout["next_states"], dtype=np.float32)
    if states.shape[0] == 0:
        return float("nan")
    return float(np.sum(np.abs(states[-1] - x_ref)))


def stability_error(rollout: Dict, x_ref: np.ndarray, tail_steps: int = 50) -> float:
    states = np.asarray(rollout["next_states"], dtype=np.float32)
    if states.shape[0] == 0:
        return float("nan")
    tail = tail_window(states, tail_steps)
    return float(np.mean(np.sum(np.abs(tail - x_ref[None, :]), axis=1)))


def plot_rollouts_xz(rollouts: List[Dict], out_path: str):
    fig, ax = plt.subplots(figsize=(7, 5))

    for i, r in enumerate(rollouts):
        s = r["states"]
        if len(s) == 0:
            continue

        x = s[:, 0]
        z_up = -s[:, 2]  # env uses positive-down pz

        label = f"traj {i}"
        if r["crashed"]:
            label += " crashed"
        if r["violated_trajectory"]:
            label += " violated"

        line, = ax.plot(x, z_up, lw=2, label=label)
        c = line.get_color()
        ax.scatter(x[0], z_up[0], color=c, marker="o", s=25)
        ax.scatter(x[-1], z_up[-1], color=c, marker="x", s=45)

    ax.axhline(0.0, color="black", linestyle="--", lw=1.5, label="ground pz=0")
    ax.set_xlabel("x")
    ax.set_ylabel("height = -pz")
    ax.set_title("CAL Quad3D rollouts: x-height")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_rollouts_xy(rollouts: List[Dict], out_path: str, safe_radius: float = 3.0):
    fig, ax = plt.subplots(figsize=(6, 6))

    for i, r in enumerate(rollouts):
        s = r["states"]
        if len(s) == 0:
            continue

        label = f"traj {i}"
        if r["crashed"]:
            label += " crashed"
        if r["violated_trajectory"]:
            label += " violated"

        line, = ax.plot(s[:, 0], s[:, 1], lw=2, label=label)
        c = line.get_color()
        ax.scatter(s[0, 0], s[0, 1], color=c, marker="o", s=25)
        ax.scatter(s[-1, 0], s[-1, 1], color=c, marker="x", s=45)

    circle = plt.Circle((0, 0), safe_radius, fill=False, linestyle="--", lw=1.5)
    ax.add_patch(circle)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("CAL Quad3D rollouts: x-y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_rollouts_3d(rollouts: List[Dict], out_path: str):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    for i, r in enumerate(rollouts):
        s = r["states"]
        if len(s) == 0:
            continue

        z_up = -s[:, 2]

        label = f"traj {i}"
        if r["crashed"]:
            label += " crashed"
        if r["violated_trajectory"]:
            label += " violated"

        line, = ax.plot(s[:, 0], s[:, 1], z_up, lw=2, label=label)
        c = line.get_color()
        ax.scatter(s[0, 0], s[0, 1], z_up[0], color=c, marker="o", s=25)
        ax.scatter(s[-1, 0], s[-1, 1], z_up[-1], color=c, marker="x", s=45)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("height = -pz")
    ax.set_title("CAL Quad3D rollouts")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True, help="Directory containing actor_*, critics_*, safetycritics_*")
    p.add_argument("--suffix", default=None, help="Checkpoint suffix. If omitted, picks latest actor_* by mtime.")
    p.add_argument("--out_dir", default="eval_cal_quad3d")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_eval", type=int, default=100)
    p.add_argument("--init_npz", default=None, help="Optional .npz file containing initial Quad3D states.")
    p.add_argument("--init_key", default="init_states", help="Array key inside --init_npz. Expected shape: (N, 9).")

    p.add_argument("--cost_lim", type=float, default=10.0)
    p.add_argument("--epoch_length", type=int, default=500)

    p.add_argument("--x_range", type=float, nargs=2, default=[-1.5, 1.5])
    p.add_argument("--y_range", type=float, nargs=2, default=[-1.5, 1.5])
    p.add_argument("--z_range", type=float, nargs=2, default=[-1.0, -0.05])

    p.add_argument("--normalize_obs", action="store_true")
    p.add_argument("--stability_tail_steps", type=int, default=50)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    actor_path, critics_path, safetycritics_path, suffix = resolve_model_paths(
        args.model_dir,
        args.suffix,
    )

    env = Quad3DEnv(
        seed=args.seed,
        max_episode_steps=args.epoch_length,
        normalize_obs=args.normalize_obs,
    )

    stats_path = os.path.join(args.model_dir, f"obs_stats_{suffix}.npz")
    obs_stats_loaded = False
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        env.set_obs_stats(stats["mean"], stats["var"], int(stats["count"]))
        obs_stats_loaded = True
        print(f"Loaded obs stats from {stats_path}")
    else:
        print(f"Warning: no obs stats at {stats_path}, normalizer starts fresh")

    env.set_eval_mode()

    cal_args = make_cal_args(cost_lim=args.cost_lim, epoch_length=args.epoch_length)

    agent = CALAgent(env.observation_space.shape[0], env.action_space, cal_args)
    agent.load_model(
        actor_path=actor_path,
        critics_path=critics_path,
        safetycritics_path=safetycritics_path,
    )
    agent.train(False)

    rng = np.random.default_rng(args.seed)
    if args.init_npz is not None:
        init_states = load_init_states(args.init_npz, args.init_key, args.n_eval)
    else:
        init_states = [
            sample_valid_init(
                rng,
                x_range=tuple(args.x_range),
                y_range=tuple(args.y_range),
                z_range=tuple(args.z_range),
            )
            for _ in range(args.n_eval)
        ]

    rollouts = []
    for i, init in enumerate(init_states):
        print(f"rollout {i + 1}/{len(init_states)}")
        rollouts.append(rollout_one(env, agent, init, seed=args.seed + i))

    x_ref = quad3d_ref(env)
    returns = np.asarray([r["return"] for r in rollouts], dtype=np.float32)
    cost_sums = np.asarray([r["cost_sum"] for r in rollouts], dtype=np.float32)
    cost_rates = np.asarray([r["cost_rate"] for r in rollouts], dtype=np.float32)

    lengths = np.asarray([r["length"] for r in rollouts], dtype=np.float32)
    crashed = np.asarray([r["crashed"] for r in rollouts], dtype=np.float32)
    terminated = np.asarray([r["terminated"] for r in rollouts], dtype=np.float32)
    truncated = np.asarray([r["truncated"] for r in rollouts], dtype=np.float32)
    ground_crashed = np.asarray([r["ground_crash"] for r in rollouts], dtype=np.float32)
    radius_crashed = np.asarray([r["radius_crash"] for r in rollouts], dtype=np.float32)

    unsafe_steps = np.asarray([r["unsafe_steps"] for r in rollouts], dtype=np.float32)
    violated_trajectories = np.asarray([r["violated_trajectory"] for r in rollouts], dtype=np.float32)
    terminal_state_errors = np.asarray(
        [terminal_state_error(r, x_ref) for r in rollouts],
        dtype=np.float32,
    )
    stability_errors = np.asarray(
        [stability_error(r, x_ref, tail_steps=args.stability_tail_steps) for r in rollouts],
        dtype=np.float32,
    )

    terminal_state_norms = np.asarray([r["terminal_state_norm"] for r in rollouts], dtype=np.float32)
    terminal_pos_norms = np.asarray([r["terminal_pos_norm"] for r in rollouts], dtype=np.float32)
    terminal_vel_norms = np.asarray([r["terminal_vel_norm"] for r in rollouts], dtype=np.float32)
    terminal_ang_norms = np.asarray([r["terminal_ang_norm"] for r in rollouts], dtype=np.float32)
    terminal_pzs = np.asarray([r["terminal_pz"] for r in rollouts], dtype=np.float32)
    terminal_hs = np.asarray([r["terminal_h"] for r in rollouts], dtype=np.float32)

    total_unsafe_steps = int(np.sum(unsafe_steps))
    total_steps = int(np.sum(lengths))

    violation_rate_per_timestep = float(total_unsafe_steps / max(total_steps, 1))
    violated_trajectory_rate = float(np.mean(violated_trajectories))
    stability_rate = float(np.nanmean(stability_errors))

    summary = {
        "n_eval": int(len(init_states)),

        "return_mean": float(np.nanmean(returns)),
        "return_std": float(np.nanstd(returns)),

        # Cost diagnostics. Cost may include penalties and is NOT the pure violation metric.
        "cost_sum_mean": float(np.nanmean(cost_sums)),
        "cost_sum_std": float(np.nanstd(cost_sums)),
        "cost_rate_mean": float(np.nanmean(cost_rates)),
        "cost_rate_std": float(np.nanstd(cost_rates)),

        # Requested safety metrics.
        "violation_rate_per_timestep": violation_rate_per_timestep,
        "violated_trajectory_rate": violated_trajectory_rate,
        "violated_trajectory_count": int(np.sum(violated_trajectories)),
        "total_unsafe_steps": total_unsafe_steps,
        "total_steps": total_steps,
        "terminal_state_error_mean": float(np.nanmean(terminal_state_errors)),
        "stability_rate": stability_rate,
        "stability_tail_steps": int(args.stability_tail_steps),

        "episode_length_mean": float(np.nanmean(lengths)),
        "episode_length_std": float(np.nanstd(lengths)),

        "crash_rate": float(np.nanmean(crashed)),
        "crash_count": int(np.nansum(crashed)),
        "ground_crash_rate": float(np.nanmean(ground_crashed)),
        "radius_crash_rate": float(np.nanmean(radius_crashed)),
        "termination_rate": float(np.nanmean(terminated)),
        "timeout_rate": float(np.nanmean(truncated)),

        "terminal_state_norm_mean": float(np.nanmean(terminal_state_norms)),
        "terminal_pos_norm_mean": float(np.nanmean(terminal_pos_norms)),
        "terminal_vel_norm_mean": float(np.nanmean(terminal_vel_norms)),
        "terminal_ang_norm_mean": float(np.nanmean(terminal_ang_norms)),
        "terminal_pz_mean": float(np.nanmean(terminal_pzs)),
        "terminal_h_mean": float(np.nanmean(terminal_hs)),

        "model_dir": args.model_dir,
        "suffix": suffix,
        "actor_path": actor_path,
        "obs_stats_path": stats_path,
        "obs_stats_loaded": obs_stats_loaded,
        "normalize_obs": bool(args.normalize_obs),
        "init_npz": args.init_npz,
        "init_key": args.init_key,
    }

    print(json.dumps(summary, indent=2))

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    rows = []
    for i, r in enumerate(rollouts):
        rows.append(
            {
                "idx": i,
                "return": r["return"],
                "cost_sum": r["cost_sum"],
                "cost_rate": r["cost_rate"],

                "unsafe_steps": r["unsafe_steps"],
                "violated_trajectory": r["violated_trajectory"],
                "terminal_state_error": float(terminal_state_errors[i]),
                "stability_error": float(stability_errors[i]),

                "length": r["length"],
                "crashed": r["crashed"],
                "ground_crash": r["ground_crash"],
                "radius_crash": r["radius_crash"],
                "terminated": r["terminated"],
                "truncated": r["truncated"],

                "terminal_state_norm": r["terminal_state_norm"],
                "terminal_pos_norm": r["terminal_pos_norm"],
                "terminal_vel_norm": r["terminal_vel_norm"],
                "terminal_ang_norm": r["terminal_ang_norm"],
                "terminal_pz": r["terminal_pz"],
                "terminal_h": r["terminal_h"],

                **{f"init_{k}": v for k, v in r["init"].items()},
            }
        )

    with open(out_dir / "per_rollout.json", "w") as f:
        json.dump(rows, f, indent=2)

    np.savez(
        out_dir / "rollouts.npz",
        init_states=np.asarray(
            [
                [
                    init["init_px"],
                    init["init_py"],
                    init["init_pz"],
                    init["init_vx"],
                    init["init_vy"],
                    init["init_vz"],
                    init["init_phi"],
                    init["init_theta"],
                    init["init_psi"],
                ]
                for init in init_states
            ],
            dtype=np.float32,
        ),
        states=np.array([r["states"] for r in rollouts], dtype=object),
        next_states=np.array([r["next_states"] for r in rollouts], dtype=object),
        actions=np.array([r["actions"] for r in rollouts], dtype=object),
        rewards=np.array([r["rewards"] for r in rollouts], dtype=object),
        costs=np.array([r["costs"] for r in rollouts], dtype=object),
        h=np.array([r["h"] for r in rollouts], dtype=object),

        returns=returns,
        cost_sums=cost_sums,
        cost_rates=cost_rates,

        unsafe_steps=unsafe_steps,
        violated_trajectories=violated_trajectories,
        violation_rate_per_timestep=np.float32(violation_rate_per_timestep),
        violated_trajectory_rate=np.float32(violated_trajectory_rate),
        terminal_state_error_mean=np.float32(np.nanmean(terminal_state_errors)),
        stability_rate=np.float32(stability_rate),
        terminal_state_errors=terminal_state_errors,
        stability_errors=stability_errors,

        lengths=lengths,
        crashed=crashed,
        ground_crashed=ground_crashed,
        radius_crashed=radius_crashed,
        terminated=terminated,
        truncated=truncated,

        terminal_state_norms=terminal_state_norms,
        terminal_pos_norms=terminal_pos_norms,
        terminal_vel_norms=terminal_vel_norms,
        terminal_ang_norms=terminal_ang_norms,
        terminal_pzs=terminal_pzs,
        terminal_hs=terminal_hs,
    )

    plot_subset = rollouts[: min(args.n_eval, 20)]
    plot_rollouts_xz(plot_subset, str(out_dir / "rollouts_xz.png"))
    plot_rollouts_xy(plot_subset, str(out_dir / "rollouts_xy.png"))
    plot_rollouts_3d(plot_subset, str(out_dir / "rollouts_3d.png"))

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
