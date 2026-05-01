#!/usr/bin/env python3
"""Rollout and plot CAL F16 stabilize-avoid policies."""

from __future__ import annotations

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

from agent.cal import CALAgent
from arguments import readParser
from env.f16 import (
    IDX_ALPHA,
    IDX_H,
    IDX_PE,
    IDX_PN,
    IDX_THETA,
    INIT_KEY_TO_IDX,
    STATE_DIM,
    F16StabilizeEnv,
    nominal_state_v5,
    perturbation_grid_init_states,
    state_to_options,
)


def resolve_model_paths(model_dir: str, suffix: str | None):
    if suffix is None:
        candidates = [f for f in os.listdir(model_dir) if f.startswith("actor_")]
        if not candidates:
            raise FileNotFoundError(f"No actor_* files found in {model_dir}")
        latest = max(candidates, key=lambda f: os.path.getmtime(os.path.join(model_dir, f)))
        suffix = latest[len("actor_") :]
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
    args.env_name = "F16"
    args.safetygym = True
    args.constraint_type = "safetygym"
    args.cost_lim = float(cost_lim)
    args.epoch_length = int(epoch_length)
    return args


def load_init_states(path: str, key: str, n_eval: int | None):
    data = np.load(path)
    if key not in data:
        raise KeyError(f"{path} missing key={key!r}. Available: {list(data.keys())}")
    states = np.asarray(data[key], dtype=np.float64)
    if states.ndim != 2 or states.shape[1] < STATE_DIM:
        raise ValueError(f"Expected {key!r} shape (N, >=16), got {states.shape}")
    if n_eval is not None:
        states = states[:n_eval]
    return states


def options_to_state(init: Dict[str, float]) -> np.ndarray:
    x = nominal_state_v5()
    for key, idx in INIT_KEY_TO_IDX.items():
        if key in init:
            x[idx] = float(init[key])
    return x


def sample_diag_init_states(env: F16StabilizeEnv, n_eval: int, seed: int, safe_margin_max: float):
    states = env.sample_x0_eval_diag_v5(num=n_eval, seed=seed, safe_margin_max=safe_margin_max)
    if len(states) >= n_eval:
        return states[:n_eval]
    return perturbation_grid_init_states(n_eval)


def rollout_one(env: F16StabilizeEnv, agent: CALAgent, init_state: np.ndarray, seed: int):
    obs, _ = env.reset(seed=seed, options=state_to_options(init_state))
    env._x = np.asarray(init_state, dtype=np.float64).copy()
    obs = env._get_obs(env._x)

    states, next_states, actions, rewards, costs, hs, goal_flags = [], [], [], [], [], [], []
    terminated = False
    truncated = False
    infos = []

    while not (terminated or truncated):
        raw_state = env.state.copy()
        action = np.asarray(agent.select_action(obs, eval=True), dtype=np.float32)
        action = np.clip(action, env.action_space.low, env.action_space.high)

        states.append(raw_state)
        actions.append(action)
        obs, reward, cost, terminated, truncated, info = env.step(action)

        next_states.append(env.state.copy())
        rewards.append(float(reward))
        costs.append(float(cost))
        hs.append(float(info.get("h", np.nan)))
        goal_flags.append(bool(info.get("in_goal", False)))
        infos.append(dict(info))

    states = np.asarray(states, dtype=np.float32)
    next_states = np.asarray(next_states, dtype=np.float32)
    h_arr = np.asarray(hs, dtype=np.float32)
    unsafe_steps = int(np.sum(h_arr > 0.0))
    terminal = next_states[-1] if len(next_states) else np.zeros(STATE_DIM, dtype=np.float32)
    return {
        "init": np.asarray(init_state, dtype=np.float32),
        "states": states,
        "next_states": next_states,
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "costs": np.asarray(costs, dtype=np.float32),
        "h": h_arr,
        "goal_flags": np.asarray(goal_flags, dtype=bool),
        "return": float(np.sum(rewards)),
        "cost_sum": float(np.sum(costs)),
        "cost_rate": float(np.mean(costs)) if costs else float("nan"),
        "unsafe_steps": unsafe_steps,
        "violated_trajectory": bool(unsafe_steps > 0),
        "length": int(len(rewards)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": any(bool(info.get("success", False)) for info in infos),
        "crashed": any(bool(info.get("crashed", False)) for info in infos),
        "terminal_h": float(env._task_safety_margin(terminal)),
        "terminal_alt": float(terminal[IDX_H]),
    }


def compute_summary(rollouts: List[Dict], args, suffix: str, actor_path: str, obs_stats_path: str, obs_stats_loaded: bool):
    returns = np.asarray([r["return"] for r in rollouts], dtype=np.float32)
    cost_sums = np.asarray([r["cost_sum"] for r in rollouts], dtype=np.float32)
    lengths = np.asarray([r["length"] for r in rollouts], dtype=np.float32)
    unsafe_steps = np.asarray([r["unsafe_steps"] for r in rollouts], dtype=np.float32)
    violated = np.asarray([r["violated_trajectory"] for r in rollouts], dtype=np.float32)
    crashed = np.asarray([r["crashed"] for r in rollouts], dtype=np.float32)
    success = np.asarray([r["success"] for r in rollouts], dtype=np.float32)
    total_steps = int(np.sum(lengths))
    total_unsafe = int(np.sum(unsafe_steps))
    return {
        "n_eval": int(len(rollouts)),
        "return_mean": float(np.nanmean(returns)),
        "return_std": float(np.nanstd(returns)),
        "cost_sum_mean": float(np.nanmean(cost_sums)),
        "cost_sum_std": float(np.nanstd(cost_sums)),
        "violation_rate_per_timestep": float(total_unsafe / max(total_steps, 1)),
        "violated_trajectory_rate": float(np.nanmean(violated)),
        "violated_trajectory_count": int(np.sum(violated)),
        "total_unsafe_steps": total_unsafe,
        "total_steps": total_steps,
        "episode_length_mean": float(np.nanmean(lengths)),
        "episode_length_std": float(np.nanstd(lengths)),
        "crash_rate": float(np.nanmean(crashed)),
        "success_rate": float(np.nanmean(success)),
        "model_dir": args.model_dir,
        "suffix": suffix,
        "actor_path": actor_path,
        "obs_stats_path": obs_stats_path,
        "obs_stats_loaded": bool(obs_stats_loaded),
        "init_npz": args.init_npz,
        "init_key": args.init_key,
    }


def plot_summary(env: F16StabilizeEnv, rollouts: List[Dict], out_path: str, best_n: int):
    show = rollouts[: min(best_n, len(rollouts))]
    if not show:
        return
    fig = plt.figure(figsize=(12, 8))

    ax_alt = fig.add_subplot(221)
    ax_alt.axhspan(env.safe_h_min, env.safe_h_max, color="#DDF2DD", alpha=0.45, label="safe")
    ax_alt.axhspan(env.goal_h_min, env.goal_h_max, color="#7BC96F", alpha=0.35, label="goal")
    for i, r in enumerate(show):
        s = r["states"]
        t = np.arange(len(s)) * env.dt
        ax_alt.plot(t, s[:, IDX_H], lw=1.4, label=f"ep {i}")
    ax_alt.set_xlabel("Time (s)")
    ax_alt.set_ylabel("H (ft)")
    ax_alt.set_title("Altitude")
    ax_alt.grid(alpha=0.25)
    ax_alt.legend(fontsize=7)

    ax_h = fig.add_subplot(222)
    ax_h.axhline(0.0, color="black", linestyle="--", lw=1.2)
    for r in show:
        h = r["h"]
        t = np.arange(len(h)) * env.dt
        ax_h.plot(t, h, lw=1.4)
    ax_h.set_xlabel("Time (s)")
    ax_h.set_ylabel("h margin")
    ax_h.set_title("Safety margin")
    ax_h.grid(alpha=0.25)

    ax_pe = fig.add_subplot(223)
    ax_pe.axhspan(-env.safe_pe, env.safe_pe, color="#DDEBFF", alpha=0.5)
    for i, r in enumerate(show):
        s = r["states"]
        label = f"ep {i}"
        if r["crashed"]:
            label += " crash"
        ax_pe.plot(s[:, IDX_PN], s[:, IDX_PE], lw=1.3, label=label)
        ax_pe.scatter(s[0, IDX_PN], s[0, IDX_PE], color="black", s=12, marker="s")
    ax_pe.set_xlabel("PN (ft)")
    ax_pe.set_ylabel("PE (ft)")
    ax_pe.set_title("Lateral corridor")
    ax_pe.grid(alpha=0.25)
    ax_pe.legend(fontsize=7)

    ax_pitch = fig.add_subplot(224)
    ax_pitch.axhline(env.safe_theta, color="black", lw=1.0, linestyle="--")
    ax_pitch.axhline(-env.safe_theta, color="black", lw=1.0, linestyle="--")
    for r in show:
        s = r["states"]
        t = np.arange(len(s)) * env.dt
        ax_pitch.plot(t, s[:, IDX_THETA], lw=1.2)
        ax_pitch.plot(t, s[:, IDX_ALPHA], lw=1.0, alpha=0.55, linestyle="--")
    ax_pitch.set_xlabel("Time (s)")
    ax_pitch.set_ylabel("rad")
    ax_pitch.set_title("Theta (solid) and alpha (dashed)")
    ax_pitch.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_3d(rollouts: List[Dict], out_path: str, best_n: int):
    show = rollouts[: min(best_n, len(rollouts))]
    if not show:
        return
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    for i, r in enumerate(show):
        s = r["states"]
        label = f"ep {i}"
        if r["crashed"]:
            label += " crash"
        ax.plot(s[:, IDX_PN], s[:, IDX_PE], s[:, IDX_H], lw=1.2, label=label)
        ax.scatter(s[0, IDX_PN], s[0, IDX_PE], s[0, IDX_H], color="black", s=12, marker="s")
    ax.set_xlabel("PN (ft)")
    ax.set_ylabel("PE (ft)")
    ax.set_zlabel("H (ft)")
    ax.set_title("CAL F16 rollouts")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--suffix", default=None)
    p.add_argument("--out_dir", default="eval_cal_f16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_eval", type=int, default=100)
    p.add_argument("--init_source", choices=["diag", "perturbations"], default="diag")
    p.add_argument("--safe_margin_max", type=float, default=-0.2)
    p.add_argument("--init_npz", default=None)
    p.add_argument("--init_key", default="init_states")
    p.add_argument("--cost_lim", type=float, default=10.0)
    p.add_argument("--epoch_length", type=int, default=640)
    p.add_argument("--best_n", type=int, default=12)
    p.add_argument("--no_plots", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    actor_path, critics_path, safetycritics_path, suffix = resolve_model_paths(args.model_dir, args.suffix)
    env = F16StabilizeEnv(seed=args.seed, max_episode_steps=args.epoch_length)
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

    cal_args = make_cal_args(args.cost_lim, args.epoch_length)
    agent = CALAgent(env.observation_space.shape[0], env.action_space, cal_args)
    agent.load_model(actor_path, critics_path, safetycritics_path)
    agent.train(False)

    if args.init_npz is not None:
        init_states = load_init_states(args.init_npz, args.init_key, args.n_eval)
    elif args.init_source == "perturbations":
        init_states = perturbation_grid_init_states(args.n_eval)
    else:
        init_states = sample_diag_init_states(env, args.n_eval, args.seed, args.safe_margin_max)

    rollouts = []
    for i, init_state in enumerate(init_states):
        print(f"rollout {i + 1}/{len(init_states)}")
        rollouts.append(rollout_one(env, agent, init_state, seed=args.seed + i))

    summary = compute_summary(rollouts, args, suffix, actor_path, stats_path, obs_stats_loaded)
    print(json.dumps(summary, indent=2))
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "per_rollout.json", "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "idx": i,
                    "return": r["return"],
                    "cost_sum": r["cost_sum"],
                    "unsafe_steps": r["unsafe_steps"],
                    "violated_trajectory": r["violated_trajectory"],
                    "length": r["length"],
                    "success": r["success"],
                    "crashed": r["crashed"],
                    "terminated": r["terminated"],
                    "truncated": r["truncated"],
                    "terminal_h": r["terminal_h"],
                    "terminal_alt": r["terminal_alt"],
                    "init_h": float(r["init"][IDX_H]),
                    "init_theta": float(r["init"][IDX_THETA]),
                }
                for i, r in enumerate(rollouts)
            ],
            f,
            indent=2,
        )

    np.savez(
        out_dir / "rollouts.npz",
        init_states=np.asarray([r["init"] for r in rollouts], dtype=np.float32),
        states=np.asarray([r["states"] for r in rollouts], dtype=object),
        next_states=np.asarray([r["next_states"] for r in rollouts], dtype=object),
        actions=np.asarray([r["actions"] for r in rollouts], dtype=object),
        rewards=np.asarray([r["rewards"] for r in rollouts], dtype=object),
        costs=np.asarray([r["costs"] for r in rollouts], dtype=object),
        h=np.asarray([r["h"] for r in rollouts], dtype=object),
        returns=np.asarray([r["return"] for r in rollouts], dtype=np.float32),
        cost_sums=np.asarray([r["cost_sum"] for r in rollouts], dtype=np.float32),
        lengths=np.asarray([r["length"] for r in rollouts], dtype=np.int32),
        success=np.asarray([r["success"] for r in rollouts], dtype=bool),
        crashed=np.asarray([r["crashed"] for r in rollouts], dtype=bool),
        terminated=np.asarray([r["terminated"] for r in rollouts], dtype=bool),
        truncated=np.asarray([r["truncated"] for r in rollouts], dtype=bool),
    )

    if not args.no_plots:
        plot_summary(env, rollouts, str(out_dir / "rollouts.png"), best_n=args.best_n)
        plot_3d(rollouts, str(out_dir / "rollouts_3d.png"), best_n=args.best_n)
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()

