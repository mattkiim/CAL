#!/usr/bin/env python3
"""CAL training entry point for the F16 stabilize-avoid task."""

from __future__ import annotations

import os
import socket
import time
from pathlib import Path

import numpy as np
import setproctitle
import torch
import wandb

from agent.cal import CALAgent
from agent.replay_memory import ReplayMemory
from arguments import readParser
from env.constraints import get_threshold
from env.f16 import F16StabilizeEnv, IDX_H, IDX_THETA, perturbation_grid_init_states, state_to_options


class F16Sampler:
    def __init__(self, env: F16StabilizeEnv):
        self.env = env
        self.current_state = None
        self.needs_reset = True

    def _reset(self):
        obs, info = self.env.reset()
        self.current_state = obs
        self.needs_reset = False
        return obs, info

    def _step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        return obs, float(reward), float(cost), bool(terminated), bool(truncated), done, info

    def sample(self, agent: CALAgent, eval_t: bool = False):
        if self.needs_reset or self.current_state is None:
            self._reset()
        cur_state = self.current_state
        action = np.asarray(agent.select_action(cur_state, eval=eval_t), dtype=np.float32)
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        next_state, reward, cost, terminated, truncated, done, info = self._step(action)
        reward_vec = np.asarray([reward, cost], dtype=np.float32)
        if done:
            self.current_state = None
            self.needs_reset = True
        else:
            self.current_state = next_state
            self.needs_reset = False
        return cur_state, action, next_state, reward_vec, done, info


def make_f16_args(args):
    args.env_name = "F16"
    args.safetygym = True
    args.constraint_type = "safetygym"
    if not hasattr(args, "f16_eval_interval") or args.f16_eval_interval is None:
        args.f16_eval_interval = 500
    if not hasattr(args, "f16_init_action_std") or args.f16_init_action_std is None:
        args.f16_init_action_std = 0.05
    if getattr(args, "f16_batch_size", None) is not None:
        args.policy_train_batch_size = int(args.f16_batch_size)
    elif args.policy_train_batch_size == 12:
        args.policy_train_batch_size = 256
    if getattr(args, "f16_num_train_repeat", None) is not None:
        args.num_train_repeat = int(args.f16_num_train_repeat)
    elif args.num_train_repeat == 10:
        args.num_train_repeat = 2
    if getattr(args, "f16_min_pool_size", None) is not None:
        args.min_pool_size = int(args.f16_min_pool_size)
    elif args.min_pool_size == 1000:
        args.min_pool_size = 5000
    return args


def evaluate_f16_grid(env: F16StabilizeEnv, agent: CALAgent, seed: int = 0):
    was_updating = getattr(env, "_update_stats", None)
    env.set_eval_mode()

    returns, costs, lengths = [], [], []
    unsafe_steps, violated, successes, crashes = [], [], [], []
    terminal_hs, terminal_alts = [], []
    init_states = perturbation_grid_init_states(8)

    try:
        for i, init_state in enumerate(init_states):
            obs, _ = env.reset(seed=seed + i, options=state_to_options(init_state))
            done = False
            ep_ret = 0.0
            ep_cost = 0.0
            steps = 0
            hs = []
            success = False
            crashed = False
            while not done and steps < env.max_episode_steps:
                action = np.asarray(agent.select_action(obs, eval=True), dtype=np.float32)
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, reward, cost, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                ep_ret += float(reward)
                ep_cost += float(cost)
                hs.append(float(info.get("h", np.nan)))
                success = success or bool(info.get("success", False))
                crashed = crashed or bool(info.get("crashed", False))
                steps += 1

            h_arr = np.asarray(hs, dtype=np.float32)
            returns.append(ep_ret)
            costs.append(ep_cost)
            lengths.append(steps)
            unsafe_steps.append(float(np.sum(h_arr > 0.0)) if len(h_arr) else 0.0)
            violated.append(float(np.any(h_arr > 0.0)) if len(h_arr) else 0.0)
            successes.append(float(success))
            crashes.append(float(crashed))
            terminal_hs.append(float(h_arr[-1]) if len(h_arr) else float("nan"))
            terminal_alts.append(float(env.state[IDX_H]))
    finally:
        if was_updating:
            env.set_train_mode()
        else:
            env.set_eval_mode()

    returns = np.asarray(returns, dtype=np.float32)
    costs = np.asarray(costs, dtype=np.float32)
    lengths = np.asarray(lengths, dtype=np.float32)
    unsafe_steps = np.asarray(unsafe_steps, dtype=np.float32)
    violated = np.asarray(violated, dtype=np.float32)
    successes = np.asarray(successes, dtype=np.float32)
    crashes = np.asarray(crashes, dtype=np.float32)
    terminal_hs = np.asarray(terminal_hs, dtype=np.float32)
    terminal_alts = np.asarray(terminal_alts, dtype=np.float32)

    return {
        "eval/return_mean": float(np.nanmean(returns)),
        "eval/return_std": float(np.nanstd(returns)),
        "eval/cost_sum_mean": float(np.nanmean(costs)),
        "eval/cost_sum_std": float(np.nanstd(costs)),
        "eval/cost_rate_mean": float(np.nanmean(costs / np.maximum(lengths, 1.0))),
        "eval/unsafe_steps_mean": float(np.nanmean(unsafe_steps)),
        "eval/violated_trajectory_rate": float(np.nanmean(violated)),
        "eval/ep_len_mean": float(np.nanmean(lengths)),
        "eval/ep_len_std": float(np.nanstd(lengths)),
        "eval/success_rate": float(np.nanmean(successes)),
        "eval/crash_rate": float(np.nanmean(crashes)),
        "eval/terminal_h_mean": float(np.nanmean(terminal_hs)),
        "eval/terminal_alt_mean": float(np.nanmean(terminal_alts)),
    }


def select_f16_wandb_metrics(eval_metrics, train_metrics):
    eval_keys = [
        "eval/return_mean",
        "eval/cost_sum_mean",
        "eval/cost_rate_mean",
        "eval/unsafe_steps_mean",
        "eval/violated_trajectory_rate",
        "eval/ep_len_mean",
        "eval/success_rate",
        "eval/crash_rate",
        "eval/terminal_h_mean",
        "eval/terminal_alt_mean",
    ]
    train_keys = [
        "train/critic_loss",
        "train/safety_critic_loss",
        "train/target_q_abs_mean",
        "train/q_abs_mean",
        "train/target_qc_mean",
        "train/target_qc_abs_mean",
        "train/qc_mean",
        "train/qc_abs_mean",
        "train/actor_loss",
        "train/alpha",
        "train/lam",
        "train/cost_batch_mean",
        "train/reward_batch_mean",
        "train/batch_terminal_rate",
        "train/batch_action_abs_mean",
    ]

    selected = {
        key: eval_metrics[key]
        for key in eval_keys
        if key in eval_metrics
    }
    selected.update(
        {
            key: train_metrics[key]
            for key in train_keys
            if key in train_metrics
        }
    )
    return selected


def exploration_before_start(args, sampler: F16Sampler, pool: ReplayMemory):
    for _ in range(args.init_exploration_steps):
        if sampler.needs_reset or sampler.current_state is None:
            sampler._reset()
        cur_state = sampler.current_state
        action = np.random.normal(
            loc=0.0,
            scale=float(args.f16_init_action_std),
            size=sampler.env.action_space.shape,
        ).astype(np.float32)
        action = np.clip(action, sampler.env.action_space.low, sampler.env.action_space.high)
        next_state, reward, cost, terminated, truncated, done, info = sampler._step(action)
        pool.push(cur_state, action, np.asarray([reward, cost], dtype=np.float32), next_state, terminated)
        if done:
            sampler.current_state = None
            sampler.needs_reset = True
        else:
            sampler.current_state = next_state
            sampler.needs_reset = False


def train_policy_repeats(args, total_step: int, train_step: int, pool: ReplayMemory, agent: CALAgent):
    if total_step % args.train_every_n_steps > 0:
        return 0, {}
    if train_step > args.max_train_repeat_per_step * max(total_step, 1):
        return 0, {}
    metric_sums = {}
    metric_count = 0
    for i in range(args.num_train_repeat):
        batch = pool.sample(args.policy_train_batch_size)
        state, action, reward, next_state, done = batch
        reward = np.squeeze(reward)
        terminal_batch = np.squeeze(done).astype(bool)
        done = (~terminal_batch).astype(int)
        update_metrics = agent.update_parameters((state, action, reward, next_state, done), i)
        update_metrics["train/batch_terminal_rate"] = float(np.mean(terminal_batch))
        update_metrics["train/batch_action_abs_mean"] = float(np.mean(np.abs(action)))

        for key, value in update_metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
        metric_count += 1

    if metric_count == 0:
        return args.num_train_repeat, {}

    return args.num_train_repeat, {
        key: value / metric_count
        for key, value in metric_sums.items()
    }


def save_checkpoint(agent: CALAgent, env: F16StabilizeEnv, model_dir: Path, suffix: str):
    agent.save_model(
        suffix=suffix,
        actor_path=str(model_dir / f"actor_{suffix}"),
        critics_path=str(model_dir / f"critics_{suffix}"),
        safetycritics_path=str(model_dir / f"safetycritics_{suffix}"),
    )
    mean, var, count = env.get_obs_stats()
    np.savez(model_dir / f"obs_stats_{suffix}.npz", mean=mean, var=var, count=np.asarray(count))


def main(args):
    args = make_f16_args(args)
    torch.set_num_threads(args.n_training_threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_num
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = F16StabilizeEnv(seed=args.seed)
    args.epoch_length = int(env.max_episode_steps)
    try:
        args.cost_lim = get_threshold(args.env_name, constraint=args.constraint_type)
    except Exception:
        args.cost_lim = 10

    if not args.experiment_name or args.experiment_name == "exp":
        i = 1
        while os.path.exists(f"models/F16_exp{i}"):
            i += 1
        args.experiment_name = f"exp{i}"

    model_dir = Path("models") / f"F16_{args.experiment_name}"
    model_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path("results") / args.env_name / args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {args.experiment_name}")
    print(f"Model dir:  {model_dir}")
    print(f"Epoch len:  {args.epoch_length}")
    print(f"Cost lim:   {args.cost_lim}")
    print(f"Seed:       {args.seed}")
    print(f"Eval every: {args.f16_eval_interval} steps")
    print(f"Init action std: {args.f16_init_action_std}")
    print(f"Batch size: {args.policy_train_batch_size}")
    print(f"Train repeats: {args.num_train_repeat}")
    print(f"Min pool size: {args.min_pool_size}")
    print(f"Grad clip: {args.grad_clip_norm}")

    run = None
    if args.use_wandb:
        run = wandb.init(
            config=args,
            project="SafeRL",
            entity=args.user_name,
            notes=socket.gethostname(),
            name=f"{args.experiment_name}_{args.cuda_num}_{args.seed}",
            group=args.env_name,
            dir=str(run_dir),
            job_type="training",
            reinit=True,
        )

    setproctitle.setproctitle(f"{args.env_name}-{args.seed}")
    agent = CALAgent(env.observation_space.shape[0], env.action_space, args)
    pool = ReplayMemory(args.replay_size)
    sampler = F16Sampler(env)

    total_step = 0
    last_eval_step = 0
    train_metric_sums = {}
    train_metric_count = 0
    print("Collecting initial F16 exploration...")
    exploration_before_start(args, sampler, pool)

    print("Training...")
    for epoch in range(1, args.num_epoch + 1):
        sta = time.time()
        train_policy_steps = 0
        env.set_train_mode()
        for _ in range(args.epoch_length):
            env.set_train_step(total_step)
            cur_state, action, next_state, reward, done, info = sampler.sample(agent, eval_t=False)
            terminated = bool(info.get("terminated", done))
            pool.push(cur_state, action, reward, next_state, terminated)
            if len(pool) > args.min_pool_size:
                update_steps, train_metrics = train_policy_repeats(
                    args,
                    total_step,
                    train_policy_steps,
                    pool,
                    agent,
                )
                train_policy_steps += update_steps
                if train_metrics:
                    for key, value in train_metrics.items():
                        train_metric_sums[key] = train_metric_sums.get(key, 0.0) + float(value)
                    train_metric_count += 1
            total_step += 1

            if total_step - last_eval_step >= args.f16_eval_interval:
                train_summary = {}
                if train_metric_count:
                    train_summary = {
                        key: value / train_metric_count
                        for key, value in train_metric_sums.items()
                    }

                metrics = evaluate_f16_grid(env, agent, seed=args.seed + 12345)
                sampler.current_state = None
                sampler.needs_reset = True
                last_eval_step = total_step
                print(
                    f"epoch={epoch:04d} step={total_step} "
                    f"return={metrics['eval/return_mean']:.2f} "
                    f"cost={metrics['eval/cost_sum_mean']:.2f} "
                    f"viol={metrics['eval/violated_trajectory_rate']:.3f} "
                    f"len={metrics['eval/ep_len_mean']:.1f} "
                    f"success={metrics['eval/success_rate']:.2f} "
                    f"crash={metrics['eval/crash_rate']:.2f} "
                    f"term_h={metrics['eval/terminal_h_mean']:.3f} "
                    f"q_loss={train_summary.get('train/critic_loss', float('nan')):.3g} "
                    f"|tq|={train_summary.get('train/target_q_abs_mean', float('nan')):.3g} "
                    f"qc_loss={train_summary.get('train/safety_critic_loss', float('nan')):.3g} "
                    f"lam={train_summary.get('train/lam', float('nan')):.3g} "
                    f"time={int(time.time() - sta)}s"
                )
                if args.use_wandb:
                    log_dict = select_f16_wandb_metrics(metrics, train_summary)
                    log_dict["budget/cost_lim"] = args.cost_lim
                    log_dict["total_step"] = total_step
                    wandb.log(log_dict, step=total_step)
                train_metric_sums = {}
                train_metric_count = 0
                env.set_train_mode()

        if args.save_parameters and epoch % 50 == 0:
            save_checkpoint(agent, env, model_dir, f"{args.seed}_epoch{epoch}")

    if args.save_parameters:
        save_checkpoint(agent, env, model_dir, f"{args.seed}_final")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    parsed = readParser()
    if getattr(parsed, "seed", None) is None:
        parsed.seed = 4921
    main(parsed)
