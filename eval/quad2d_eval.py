#!/usr/bin/env python3
import argparse
import json
import os
import sys
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
from env.quad2d import Quad2DEnv


EVAL_STARTS = np.array([
    [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.53, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.47, 0.0, 0.0, 0.0],
], dtype=np.float32)

def load_init_states(path: str, key: str, num_episodes: int | None):
    data = np.load(path)
    if key not in data:
        raise KeyError(f"{path} missing key={key!r}. Available: {list(data.keys())}")
    states = np.asarray(data[key], dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 6:
        raise ValueError(f"Expected shape (N, 6), got {states.shape}")
    return states[:num_episodes] if num_episodes is not None else states


def rollout_from_init_state(env, agent, init_state, seed: int):
    obs, _ = env.reset(
        options={
            "init_x": float(init_state[0]),
            "init_vx": float(init_state[1]),
            "init_z": float(init_state[2]),
            "init_vz": float(init_state[3]),
            "init_theta": float(init_state[4]),
            "init_omega": float(init_state[5]),
        }
    )

    states, refs, actions = [], [], []
    rewards, costs = [], []
    weighted_state_errors, unweighted_state_errors = [], []
    weighted_action_errors, unweighted_action_errors = [], []

    xs = [float(env.state[0])]
    zs = [float(env.state[2])]

    terminated = False
    truncated = False

    Q = np.asarray(env.Q, dtype=np.float32)
    R = np.asarray(env.R, dtype=np.float32)
    a_ref = np.asarray(env.a_ref, dtype=np.float32)

    while not (terminated or truncated):
        state = np.asarray(env.state, dtype=np.float32)
        ref = np.asarray(env._waypoints[env._waypoint_idx], dtype=np.float32)

        action = agent.select_action(obs, eval=True)
        action = np.asarray(action, dtype=np.float32)

        state_err = state - ref
        action_raw = (np.clip(action, -1.0, 1.0) + 1.0) / 2.0
        action_err = action_raw - a_ref

        weighted_state_errors.append(float(state_err @ Q @ state_err))
        unweighted_state_errors.append(float(np.sum(state_err ** 2)))
        weighted_action_errors.append(float(action_err @ R @ action_err))
        unweighted_action_errors.append(float(np.sum(action_err ** 2)))

        states.append(state)
        refs.append(ref)
        actions.append(action)

        obs, reward, cost, terminated, truncated, info = env.step(action)

        rewards.append(float(reward))
        costs.append(float(cost))

    xs.append(float(env.state[0]))
    zs.append(float(env.state[2]))
    
    return {
        "states": np.asarray(states, dtype=np.float32),
        "refs": np.asarray(refs, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "costs": np.asarray(costs, dtype=np.float32),
        "weighted_state_errors": np.asarray(weighted_state_errors, dtype=np.float32),
        "unweighted_state_errors": np.asarray(unweighted_state_errors, dtype=np.float32),
        "weighted_action_errors": np.asarray(weighted_action_errors, dtype=np.float32),
        "unweighted_action_errors": np.asarray(unweighted_action_errors, dtype=np.float32),
        "x": np.asarray(xs, dtype=np.float32),
        "z": np.asarray(zs, dtype=np.float32),
        "terminated": bool(terminated),
    }


def plot_rollouts(rollouts: List[Dict[str, np.ndarray]], out_path: str, best_n: int = 100):
    fig, ax = plt.subplots(figsize=(7, 6))

    x_line = np.linspace(-2.2, 2.2, 200)
    ax.fill_between(x_line, -0.2, 0.5, alpha=0.25)
    ax.fill_between(x_line, 1.5, 2.2, alpha=0.25)
    ax.plot(x_line, np.full_like(x_line, 0.5), "k-", lw=2)
    ax.plot(x_line, np.full_like(x_line, 1.5), "k-", lw=2)

    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), 1 + np.sin(theta), "k--", lw=1.5, label="Reference")

    ranked = sorted(
        rollouts,
        key=lambda r: (
            bool(r["terminated"]),
            float(r["costs"].sum()),
            -float(r["rewards"].sum()),
        ),
    )

    for j, r in enumerate(ranked[:best_n]):
        ax.plot(r["x"], r["z"], lw=2.0, label="CAL" if j == 0 else None)
        ax.plot(r["x"][0], r["z"][0], "o")
        if r["terminated"]:
            ax.plot(r["x"][-1], r["z"][-1], "x", markersize=10)

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.2, 2.2)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor_path", required=True)
    parser.add_argument("--critics_path", default=None)
    parser.add_argument("--safetycritics_path", default=None)

    parser.add_argument("--init_npz", default=None)
    parser.add_argument("--four_starts", action="store_true")
    parser.add_argument("--init_key", default="init_states")
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default="eval_cal_quad2d_from_inits")

    args_cli = parser.parse_args()

    os.makedirs(args_cli.out_dir, exist_ok=True)

    # Reuse CAL argument defaults so model architecture matches training.
    import sys

    old_argv = sys.argv
    sys.argv = [sys.argv[0]]   # hide CLI args from readParser
    cal_args = readParser()
    sys.argv = old_argv

    cal_args.env_name = "Quad2D"
    cal_args.safetygym = True
    cal_args.constraint_type = "safetygym"
    cal_args.cost_lim = 10
    cal_args.epoch_length = 360

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    env = Quad2DEnv(seed=args_cli.seed)
    
    obs, _ = env.reset()
    # for _ in range(200000):
    #     action = env.action_space.sample()
    #     obs, reward, cost, terminated, truncated, _ = env.step(action)
    #     if terminated or truncated:
    #         obs, _ = env.reset()

    env.set_eval_mode()

    # env._normalize_obs = False
    s_dim = env.observation_space.shape[0]

    agent = CALAgent(s_dim, env.action_space, cal_args)
    agent.load_model(
        actor_path=args_cli.actor_path,
        critics_path=args_cli.critics_path,
        safetycritics_path=args_cli.safetycritics_path,
    )
    agent.train(False)

    if args_cli.four_starts:
        init_states = EVAL_STARTS
    elif args_cli.init_npz is not None:
        init_states = load_init_states(
            args_cli.init_npz,
            args_cli.init_key,
            args_cli.num_episodes,
        )
    else:
        raise ValueError("Provide either --init_npz or --four_starts")

    rollouts = []
    for i, init_state in enumerate(init_states):
        print(f"rollout {i + 1}/{len(init_states)}")
        rollouts.append(
            rollout_from_init_state(
                env=env,
                agent=agent,
                init_state=init_state,
                seed=args_cli.seed + i,
            )
        )

    returns = np.asarray([r["rewards"].sum() for r in rollouts], dtype=np.float32)
    cost_sums = np.asarray([r["costs"].sum() for r in rollouts], dtype=np.float32)
    violation_rates = np.asarray(
        [r["costs"].mean() if len(r["costs"]) else 0.0 for r in rollouts],
        dtype=np.float32,
    )
    lengths = np.asarray([len(r["rewards"]) for r in rollouts], dtype=np.float32)
    crashes = np.asarray([r["terminated"] for r in rollouts], dtype=np.float32)

    def has_alt_violation(r):
        z = r["states"][:, 2]
        return bool(np.any((z < 0.5) | (z > 1.5)))

    episode_alt_violation_rate = float(np.mean([has_alt_violation(r) for r in rollouts]))

    weighted_state_error_means = np.asarray(
        [r["weighted_state_errors"].mean() for r in rollouts],
        dtype=np.float32,
    )
    unweighted_state_error_means = np.asarray(
        [r["unweighted_state_errors"].mean() for r in rollouts],
        dtype=np.float32,
    )
    weighted_action_error_means = np.asarray(
        [r["weighted_action_errors"].mean() for r in rollouts],
        dtype=np.float32,
    )
    unweighted_action_error_means = np.asarray(
        [r["unweighted_action_errors"].mean() for r in rollouts],
        dtype=np.float32,
    )

    save_dict = {
        "init_states": init_states,
        "returns": returns,
        "cost_sums": cost_sums,
        "violation_rates": violation_rates,
        "lengths": lengths,
        "weighted_state_error_means": weighted_state_error_means,
        "unweighted_state_error_means": unweighted_state_error_means,
        "weighted_action_error_means": weighted_action_error_means,
        "unweighted_action_error_means": unweighted_action_error_means,
        "crash_rate": np.float32(crashes.mean()),
        "episode_alt_violation_rate": np.float32(episode_alt_violation_rate),
        "num_episodes": np.int64(len(rollouts)),
        "label": np.array("CAL"),
    }

    for i, r in enumerate(rollouts):
        save_dict[f"x_{i}"] = r["x"]
        save_dict[f"z_{i}"] = r["z"]
        save_dict[f"total_reward_{i}"] = np.float32(r["rewards"].sum())
        save_dict[f"total_cost_{i}"] = np.float32(r["costs"].sum())
        save_dict[f"terminated_{i}"] = np.bool_(r["terminated"])

    np.savez(os.path.join(args_cli.out_dir, "rollouts_from_inits.npz"), **save_dict)

    plot_rollouts(
        rollouts,
        out_path=os.path.join(args_cli.out_dir, "trajectories.png"),
        best_n=100,
    )

    summary = {
        "num_episodes": int(len(init_states)),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "cost_sum_mean": float(cost_sums.mean()),
        "cost_sum_std": float(cost_sums.std()),
        "violation_rate_mean": float(violation_rates.mean()),
        "length_mean": float(lengths.mean()),
        "weighted_state_error_mean": float(weighted_state_error_means.mean()),
        "unweighted_state_error_mean": float(unweighted_state_error_means.mean()),
        "weighted_action_error_mean": float(weighted_action_error_means.mean()),
        "unweighted_action_error_mean": float(unweighted_action_error_means.mean()),
        "crash_rate": float(crashes.mean()),
        "episode_alt_violation_rate": episode_alt_violation_rate,
        "actor_path": args_cli.actor_path,
        "init_npz": args_cli.init_npz,
        "init_key": args_cli.init_key,
    }

    with open(os.path.join(args_cli.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved outputs to: {args_cli.out_dir}")


if __name__ == "__main__":
    main()