#!/usr/bin/env python3
import os
import time
import socket
from pathlib import Path

import numpy as np
import torch
import wandb
import setproctitle

from arguments import readParser
from env.constraints import get_threshold
from agent.cal import CALAgent
from agent.replay_memory import ReplayMemory
from env.quad3d import Quad3DEnv


"""
Run example:

python train_quad3d.py \
  --num_epoch 500 \
  --save_parameters \
  --use_wandb \
  --experiment_name exp1 \
  --init_exploration_steps 10000 \
  --min_pool_size 10000 \
  --num_eval_epochs 4 \
  --lr 1e-4 \
  --qc_lr 1e-4
"""


# Same fixed starts as RAC Quad3D eval.
# State convention: [px, py, pz, vx, vy, vz, phi, theta, psi]
# pz is positive downward, so pz < 0 is above ground.
EVAL_STARTS = [
    (0.5, 0.0, -0.3),
    (-0.5, 0.0, -0.3),
    (0.0, 0.5, -0.4),
    (0.0, -0.5, -0.2),
]


class Quad3DSampler:
    """
    CAL-compatible sampler for Quad3D.

    Assumes Quad3DEnv.step(action) returns:
        obs, reward, cost, terminated, truncated, info

    Returns:
        cur_state, action, next_state, reward_vec, done, info

    where:
        reward_vec = np.array([reward, cost])
    """

    def __init__(self, env):
        self.env = env
        self.current_state = None
        self.needs_reset = True

    def _reset(self):
        out = self.env.reset()
        if isinstance(out, tuple):
            obs, info = out
        else:
            obs, info = out, {}

        self.current_state = obs
        self.needs_reset = False
        return obs, info

    def _step(self, action):
        out = self.env.step(action)

        if len(out) != 6:
            raise ValueError(f"Expected 6 returns from Quad3DEnv.step, got {len(out)}")

        obs, reward, cost, terminated, truncated, info = out

        cost = float(cost)
        terminated = bool(terminated)
        truncated = bool(truncated)
        done = bool(terminated or truncated)

        return obs, float(reward), cost, terminated, truncated, done, info

    def sample(self, agent, eval_t=False):
        if self.needs_reset or self.current_state is None:
            self._reset()

        cur_state = self.current_state
        action = agent.select_action(cur_state, eval=eval_t)

        next_state, reward, cost, terminated, truncated, done, info = self._step(action)

        reward_vec = np.array([reward, cost], dtype=np.float32)

        if done:
            self.needs_reset = True
            self.current_state = None
        else:
            self.current_state = next_state

        return cur_state, action, next_state, reward_vec, done, info


def evaluate_quad3d_four_starts(env, agent, seed=0, debug=False):
    returns = []
    cost_sums = []
    lengths = []
    crash_flags = []
    timeout_flags = []

    cost_rate_per_ep = []
    cost_rate_full_per_ep = []

    unsafe_steps_list = []
    true_violation_rate_list = []
    true_violation_rate_full_list = []

    crash_penalty_steps_list = []
    crash_penalty_sum_list = []

    mean_hs = []
    max_hs = []
    terminal_hs = []

    terminal_state_norms = []
    terminal_pos_norms = []
    terminal_vel_norms = []
    terminal_ang_norms = []
    terminal_pzs = []

    ground_crash_flags = []
    radius_crash_flags = []

    if hasattr(env, "set_eval_mode"):
        env.set_eval_mode()

    for i, (px0, py0, pz0) in enumerate(EVAL_STARTS):
        out = env.reset(
            seed=seed + i,
            options={
                "init_px": px0,
                "init_py": py0,
                "init_pz": pz0,
                "init_vx": 0.0,
                "init_vy": 0.0,
                "init_vz": 0.0,
                "init_phi": 0.0,
                "init_theta": 0.0,
                "init_psi": 0.0,
            },
        )

        if isinstance(out, tuple):
            obs, _ = out
        else:
            obs = out

        ep_ret = 0.0
        ep_cost = 0.0
        steps = 0
        done = False
        terminated = False
        truncated = False

        hs = []
        costs = []

        while not done and steps < env.max_episode_steps:
            action = agent.select_action(obs, eval=True)

            out = env.step(action)

            if len(out) != 6:
                raise ValueError(f"Expected 6 returns from Quad3DEnv.step, got {len(out)}")

            obs, reward, cost, terminated, truncated, info = out

            cost = float(cost)
            terminated = bool(terminated)
            truncated = bool(truncated)
            h_val = float(info.get("h", np.nan))

            done = bool(terminated or truncated)

            ep_ret += float(reward)
            ep_cost += cost
            costs.append(cost)
            hs.append(h_val)
            steps += 1

        state = getattr(env, "state", np.zeros(9, dtype=np.float32))

        state_norm = float(np.linalg.norm(state))
        pos_norm = float(np.linalg.norm(state[:3]))
        vel_norm = float(np.linalg.norm(state[3:6]))
        ang_norm = float(np.linalg.norm(state[6:9]))
        terminal_pz = float(state[2])
        terminal_h = float(max(terminal_pz, state_norm - 3.0))

        if debug:
            print(
                "eval end:",
                "start=", (px0, py0, pz0),
                "steps=", steps,
                "terminated=", terminated,
                "truncated=", truncated,
                "pz=", terminal_pz,
                "norm=", state_norm,
                "h=", terminal_h,
                "cost_sum=", ep_cost,
            )

        hs_arr = np.asarray(hs, dtype=np.float32)
        costs_arr = np.asarray(costs, dtype=np.float32)

        unsafe_steps = int(np.sum(hs_arr > 0.0)) if len(hs_arr) else 0

        # If env adds crash penalty, costs > 1 usually indicate that.
        crash_penalty_steps = int(np.sum(costs_arr > 1.0)) if len(costs_arr) else 0
        crash_penalty_sum = (
            float(np.sum(np.maximum(costs_arr - 1.0, 0.0))) if len(costs_arr) else 0.0
        )

        returns.append(ep_ret)
        cost_sums.append(ep_cost)
        lengths.append(steps)

        crash_flags.append(float(terminated))
        timeout_flags.append(float(truncated))

        cost_rate_per_ep.append(ep_cost / max(steps, 1))
        cost_rate_full_per_ep.append(ep_cost / float(env.max_episode_steps))

        unsafe_steps_list.append(unsafe_steps)
        true_violation_rate_list.append(unsafe_steps / max(steps, 1))
        true_violation_rate_full_list.append(unsafe_steps / float(env.max_episode_steps))

        crash_penalty_steps_list.append(crash_penalty_steps)
        crash_penalty_sum_list.append(crash_penalty_sum)

        mean_hs.append(float(np.nanmean(hs_arr)) if len(hs_arr) else float("nan"))
        max_hs.append(float(np.nanmax(hs_arr)) if len(hs_arr) else float("nan"))
        terminal_hs.append(terminal_h)

        terminal_state_norms.append(state_norm)
        terminal_pos_norms.append(pos_norm)
        terminal_vel_norms.append(vel_norm)
        terminal_ang_norms.append(ang_norm)
        terminal_pzs.append(terminal_pz)

        ground_crash_flags.append(float(terminal_pz >= 0.3))
        radius_crash_flags.append(float(state_norm >= 3.5))

    returns = np.asarray(returns, dtype=np.float32)
    cost_sums = np.asarray(cost_sums, dtype=np.float32)
    lengths = np.asarray(lengths, dtype=np.float32)
    crash_flags = np.asarray(crash_flags, dtype=np.float32)
    timeout_flags = np.asarray(timeout_flags, dtype=np.float32)

    cost_rate_per_ep = np.asarray(cost_rate_per_ep, dtype=np.float32)
    cost_rate_full_per_ep = np.asarray(cost_rate_full_per_ep, dtype=np.float32)

    unsafe_steps_list = np.asarray(unsafe_steps_list, dtype=np.float32)
    true_violation_rate_list = np.asarray(true_violation_rate_list, dtype=np.float32)
    true_violation_rate_full_list = np.asarray(true_violation_rate_full_list, dtype=np.float32)

    crash_penalty_steps_list = np.asarray(crash_penalty_steps_list, dtype=np.float32)
    crash_penalty_sum_list = np.asarray(crash_penalty_sum_list, dtype=np.float32)

    mean_hs = np.asarray(mean_hs, dtype=np.float32)
    max_hs = np.asarray(max_hs, dtype=np.float32)
    terminal_hs = np.asarray(terminal_hs, dtype=np.float32)

    terminal_state_norms = np.asarray(terminal_state_norms, dtype=np.float32)
    terminal_pos_norms = np.asarray(terminal_pos_norms, dtype=np.float32)
    terminal_vel_norms = np.asarray(terminal_vel_norms, dtype=np.float32)
    terminal_ang_norms = np.asarray(terminal_ang_norms, dtype=np.float32)
    terminal_pzs = np.asarray(terminal_pzs, dtype=np.float32)

    ground_crash_flags = np.asarray(ground_crash_flags, dtype=np.float32)
    radius_crash_flags = np.asarray(radius_crash_flags, dtype=np.float32)

    return {
        "eval/return_mean": float(np.nanmean(returns)),
        "eval/return_std": float(np.nanstd(returns)),

        # Raw CAL cost metrics: includes crash penalty if env adds it.
        "eval/cost_sum_mean": float(np.nanmean(cost_sums)),
        "eval/cost_sum_std": float(np.nanstd(cost_sums)),
        "eval/cost_rate_mean": float(np.nanmean(cost_rate_per_ep)),
        "eval/cost_rate_full_mean": float(np.nanmean(cost_rate_full_per_ep)),

        # True violation metrics: only h > 0.
        "eval/unsafe_steps_mean": float(np.nanmean(unsafe_steps_list)),
        "eval/true_violation_rate_mean": float(np.nanmean(true_violation_rate_list)),
        "eval/true_violation_rate_full_mean": float(np.nanmean(true_violation_rate_full_list)),

        # Crash penalty diagnostics.
        "eval/crash_penalty_steps_mean": float(np.nanmean(crash_penalty_steps_list)),
        "eval/crash_penalty_sum_mean": float(np.nanmean(crash_penalty_sum_list)),

        "eval/ep_len_mean": float(np.nanmean(lengths)),
        "eval/ep_len_std": float(np.nanstd(lengths)),

        "eval/crash_rate": float(np.nanmean(crash_flags)),
        "eval/timeout_rate": float(np.nanmean(timeout_flags)),
        "eval/ground_crash_rate": float(np.nanmean(ground_crash_flags)),
        "eval/radius_crash_rate": float(np.nanmean(radius_crash_flags)),

        "eval/mean_h": float(np.nanmean(mean_hs)),
        "eval/max_h": float(np.nanmean(max_hs)),
        "eval/terminal_h_mean": float(np.nanmean(terminal_hs)),

        "eval/terminal_state_norm_mean": float(np.nanmean(terminal_state_norms)),
        "eval/terminal_pos_norm_mean": float(np.nanmean(terminal_pos_norms)),
        "eval/terminal_vel_norm_mean": float(np.nanmean(terminal_vel_norms)),
        "eval/terminal_ang_norm_mean": float(np.nanmean(terminal_ang_norms)),
        "eval/terminal_pz_mean": float(np.nanmean(terminal_pzs)),
    }


def exploration_near_hover_before_start(args, sampler, pool):
    """
    Fill initial buffer with near-hover exploration.

    For Quad3D, uniform random actions often produce mostly crash data.
    Near-hover actions give the critic stable dynamics first.
    """

    for _ in range(args.init_exploration_steps):
        if sampler.needs_reset or sampler.current_state is None:
            sampler._reset()

        cur_state = sampler.current_state

        action = np.random.normal(
            loc=0.0,
            scale=args.quad3d_init_action_std,
            size=sampler.env.action_space.shape,
        ).astype(np.float32)
        action = np.clip(action, -1.0, 1.0)

        next_state, reward, cost, terminated, truncated, done, info = sampler._step(action)
        reward_vec = np.array([reward, cost], dtype=np.float32)

        pool.push(cur_state, action, reward_vec, next_state, terminated)

        if done:
            sampler.needs_reset = True
            sampler.current_state = None
        else:
            sampler.current_state = next_state
            
def exploration_before_start(args, env_sampler, pool, agent):
    for _ in range(args.init_exploration_steps):
        cur_state, action, next_state, reward, done, info = env_sampler.sample(agent, eval_t=False)
        terminated = bool(info.get("terminated", done))
        pool.push(cur_state, action, reward, next_state, terminated)


def train_policy_repeats(args, total_step, train_step, pool, agent):
    if total_step % args.train_every_n_steps > 0:
        return 0, {}

    if train_step > args.max_train_repeat_per_step * max(total_step, 1):
        return 0, {}

    metric_sums = {}
    metric_count = 0
    for i in range(args.num_train_repeat):
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = pool.sample(
            args.policy_train_batch_size
        )

        batch_reward = np.squeeze(batch_reward)
        batch_done = np.squeeze(batch_done)
        terminal_batch = batch_done.astype(bool)
        batch_done = (~terminal_batch).astype(int)

        for name, arr in [
            ("batch_state", batch_state),
            ("batch_action", batch_action),
            ("batch_reward", batch_reward),
            ("batch_next_state", batch_next_state),
            ("batch_done", batch_done),
        ]:
            if not np.all(np.isfinite(arr)):
                raise RuntimeError(f"Non-finite values in {name}")

        update_metrics = agent.update_parameters(
            (batch_state, batch_action, batch_reward, batch_next_state, batch_done),
            i,
        )
        update_metrics["train/batch_terminal_rate"] = float(np.mean(terminal_batch))
        update_metrics["train/batch_action_abs_mean"] = float(np.mean(np.abs(batch_action)))

        for key, value in update_metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
        metric_count += 1

    if metric_count == 0:
        return args.num_train_repeat, {}

    return args.num_train_repeat, {
        key: value / metric_count
        for key, value in metric_sums.items()
    }


def save_checkpoint(args, agent, env, model_dir, suffix):
    actor_path = model_dir / f"actor_{suffix}"
    critics_path = model_dir / f"critics_{suffix}"
    safetycritics_path = model_dir / f"safetycritics_{suffix}"

    agent.save_model(
        suffix=suffix,
        actor_path=str(actor_path),
        critics_path=str(critics_path),
        safetycritics_path=str(safetycritics_path),
    )

    if hasattr(env, "get_obs_stats"):
        mean, var, count = env.get_obs_stats()
        np.savez(
            model_dir / f"obs_stats_{suffix}.npz",
            mean=mean,
            var=var,
            count=np.array(count),
        )


def maybe_add_quad3d_args(args):
    """
    Add script-local defaults without editing arguments.py.
    """

    if not hasattr(args, "quad3d_eval_interval") or args.quad3d_eval_interval is None:
        args.quad3d_eval_interval = 500

    if not hasattr(args, "quad3d_init_action_std") or args.quad3d_init_action_std is None:
        args.quad3d_init_action_std = 0.05

    if not hasattr(args, "quad3d_debug_eval"):
        args.quad3d_debug_eval = False

    if getattr(args, "quad3d_batch_size", None) is not None:
        args.policy_train_batch_size = int(args.quad3d_batch_size)
    elif args.policy_train_batch_size == 12:
        args.policy_train_batch_size = 256

    if getattr(args, "quad3d_num_train_repeat", None) is not None:
        args.num_train_repeat = int(args.quad3d_num_train_repeat)
    elif args.num_train_repeat == 10:
        args.num_train_repeat = 2

    if getattr(args, "quad3d_min_pool_size", None) is not None:
        args.min_pool_size = int(args.quad3d_min_pool_size)
    elif args.min_pool_size == 1000:
        args.min_pool_size = 5000

    return args


def select_quad3d_wandb_metrics(eval_metrics, train_metrics):
    eval_keys = [
        "eval/return_mean",
        "eval/cost_sum_mean",
        "eval/true_violation_rate_mean",
        "eval/unsafe_steps_mean",
        "eval/ep_len_mean",
        "eval/crash_rate",
        "eval/ground_crash_rate",
        "eval/radius_crash_rate",
        "eval/terminal_h_mean",
        "eval/terminal_pz_mean",
        "eval/terminal_state_norm_mean",
    ]
    train_keys = [
        "train/critic_loss",
        "train/safety_critic_loss",
        "train/target_q_mean",
        "train/target_q_abs_mean",
        "train/q_mean",
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


def main(args):
    args = maybe_add_quad3d_args(args)

    torch.set_num_threads(args.n_training_threads)

    args.env_name = "Quad3D"
    args.safetygym = True
    args.constraint_type = "safetygym"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_num

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = Quad3DEnv()

    # Use env horizon directly.
    args.epoch_length = int(env.max_episode_steps)

    try:
        args.cost_lim = get_threshold(args.env_name, constraint=args.constraint_type)
    except Exception:
        args.cost_lim = 10

    s_dim = env.observation_space.shape[0]

    if not args.experiment_name or args.experiment_name == "exp":
        i = 1
        while os.path.exists(f"models/Quad3D_exp{i}"):
            i += 1
        args.experiment_name = f"exp{i}"

    model_dir = Path("models") / f"Quad3D_{args.experiment_name}"
    model_dir.mkdir(parents=True, exist_ok=True)

    run_dir = Path("results") / args.env_name / args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {args.experiment_name}")
    print(f"Model dir:  {model_dir}")
    print(f"Epoch len:  {args.epoch_length}")
    print(f"Cost lim:   {args.cost_lim}")
    print(f"Seed:       {args.seed}")
    print(f"Eval every: {args.quad3d_eval_interval} steps")
    print(f"Init action std: {args.quad3d_init_action_std}")
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

    agent = CALAgent(s_dim, env.action_space, args)
    pool = ReplayMemory(args.replay_size)
    sampler = Quad3DSampler(env)

    total_step = 0
    last_eval_step = 0
    train_metric_sums = {}
    train_metric_count = 0

    print("Collecting initial near-hover exploration...")
    exploration_near_hover_before_start(args, sampler, pool)

    print("Training...")
    for epoch in range(1, args.num_epoch + 1):
        sta = time.time()
        train_policy_steps = 0

        if hasattr(env, "set_train_mode"):
            env.set_train_mode()

        for _ in range(args.epoch_length):
            cur_state, action, next_state, reward, done, info = sampler.sample(
                agent,
                eval_t=False,
            )

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

            if total_step - last_eval_step >= args.quad3d_eval_interval:
                train_summary = {}
                if train_metric_count:
                    train_summary = {
                        key: value / train_metric_count
                        for key, value in train_metric_sums.items()
                    }

                metrics = evaluate_quad3d_four_starts(
                    env,
                    agent,
                    seed=args.seed + 12345,
                    debug=args.quad3d_debug_eval,
                )
                sampler.current_state = None
                sampler.needs_reset = True
                last_eval_step = total_step

                print(
                    f"epoch={epoch:04d} step={total_step} "
                    f"return={metrics['eval/return_mean']:.2f} "
                    f"crash={metrics['eval/crash_rate']:.2f} "
                    f"viol={metrics['eval/true_violation_rate_mean']:.3f} "
                    f"cost={metrics['eval/cost_sum_mean']:.1f} "
                    f"len={metrics['eval/ep_len_mean']:.1f} "
                    f"ground_crash={metrics['eval/ground_crash_rate']:.2f} "
                    f"radius_crash={metrics['eval/radius_crash_rate']:.2f} "
                    f"term_h={metrics['eval/terminal_h_mean']:.3f} "
                    f"q_loss={train_summary.get('train/critic_loss', float('nan')):.3g} "
                    f"|tq|={train_summary.get('train/target_q_abs_mean', float('nan')):.3g} "
                    f"qc_loss={train_summary.get('train/safety_critic_loss', float('nan')):.3g} "
                    f"lam={train_summary.get('train/lam', float('nan')):.3g} "
                    f"time={int(time.time() - sta)}s"
                )

                if args.use_wandb:
                    log_dict = select_quad3d_wandb_metrics(metrics, train_summary)
                    log_dict["budget/cost_lim"] = args.cost_lim
                    log_dict["total_step"] = total_step
                    wandb.log(log_dict, step=total_step)

                train_metric_sums = {}
                train_metric_count = 0

                if hasattr(env, "set_train_mode"):
                    env.set_train_mode()

        if args.save_parameters and epoch % 50 == 0:
            suffix = f"{args.seed}_epoch{epoch}"
            save_checkpoint(args, agent, env, model_dir, suffix)

    if args.save_parameters:
        suffix = f"{args.seed}_final"
        save_checkpoint(args, agent, env, model_dir, suffix)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    args = readParser()

    if getattr(args, "seed", None) is None:
        args.seed = 4921

    main(args)
