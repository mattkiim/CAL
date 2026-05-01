import time
import torch
import numpy as np
import os
import wandb
import socket
from pathlib import Path
import setproctitle

from arguments import readParser
from env.constraints import get_threshold
import safety_gymnasium as gym

from agent.replay_memory import ReplayMemory
from agent.cal import CALAgent
from sampler.mujoco_env_sampler import MuJoCoEnvSampler
from sampler.safetygym_env_sampler import SafetygymEnvSampler

from env.quad2d import Quad2DEnv
from env.quad3d import Quad3DEnv

EVAL_STARTS = [
    (1.0, 1.0),
    (-1.0, 1.0),
    (0.0, 0.53),
    (0.0, 1.47),
]


def evaluate_quad2d_four_starts(env, agent, seed=0):
    returns, costs, ep_lens, viol_rates = [], [], [], []
    crash_flags, timeout_flags = [], []
    was_updating_stats = getattr(env, "_update_stats", None)
    env.set_eval_mode()

    try:
        for i, (x0, z0) in enumerate(EVAL_STARTS):
            obs, _ = env.reset(
                options={
                    "init_x": x0,
                    "init_z": z0,
                    "init_vx": 0.0,
                    "init_vz": 0.0,
                    "init_theta": 0.0,
                    "init_omega": 0.0,
                }
            )

            ep_ret = 0.0
            ep_cost = 0.0
            steps = 0
            done = False
            terminated = False
            truncated = False

            while not done and steps < env.max_episode_steps:
                action = agent.select_action(obs, eval=True)
                obs, reward, cost, terminated, truncated, info = env.step(action)

                done = bool(terminated or truncated)
                ep_ret += float(reward)
                ep_cost += float(cost)
                steps += 1

            returns.append(ep_ret)
            costs.append(ep_cost)
            ep_lens.append(steps)
            viol_rates.append(ep_cost / float(env.max_episode_steps))
            crash_flags.append(float(terminated))
            timeout_flags.append(float(truncated))
    finally:
        if was_updating_stats is not None:
            if was_updating_stats:
                env.set_train_mode()
            else:
                env.set_eval_mode()

    returns = np.asarray(returns)
    costs = np.asarray(costs)
    ep_lens = np.asarray(ep_lens)
    viol_rates = np.asarray(viol_rates)
    crash_flags = np.asarray(crash_flags)
    timeout_flags = np.asarray(timeout_flags)

    return {
        "eval/return_mean": float(returns.mean()),
        "eval/return_std": float(returns.std()),
        "eval/cost_mean": float(costs.mean()),
        "eval/cost_std": float(costs.std()),
        "eval/ep_len_mean": float(ep_lens.mean()),
        "eval/ep_len_std": float(ep_lens.std()),
        "eval/violation_rate_mean": float(viol_rates.mean()),
        "eval/violation_rate_std": float(viol_rates.std()),
        "eval/crash_rate": float(crash_flags.mean()),
        "eval/timeout_rate": float(timeout_flags.mean()),
    }


def resolve_resume_paths(model_dir, suffix=None):
    """Find actor/critics/safetycritics + obs_stats in model_dir."""
    if suffix is None:
        candidates = [f for f in os.listdir(model_dir) if f.startswith("actor_")]
        if not candidates:
            raise FileNotFoundError(f"No actor_* files in {model_dir}")
        latest = max(candidates, key=lambda f: os.path.getmtime(os.path.join(model_dir, f)))
        suffix = latest[len("actor_"):]
        print(f"Auto-selected resume suffix: {suffix!r}")

    return {
        "actor": os.path.join(model_dir, f"actor_{suffix}"),
        "critics": os.path.join(model_dir, f"critics_{suffix}"),
        "safetycritics": os.path.join(model_dir, f"safetycritics_{suffix}"),
        "obs_stats": os.path.join(model_dir, f"obs_stats_{suffix}.npz"),
        "suffix": suffix,
    }


def parse_epoch_from_suffix(suffix):
    """Extract epoch number from suffix like '4921_epoch150' -> 150. Returns 0 if not found."""
    if "_epoch" in suffix:
        try:
            return int(suffix.split("_epoch")[-1])
        except ValueError:
            return 0
    return 0


def select_quad2d_wandb_metrics(eval_metrics, train_metrics):
    eval_keys = [
        "eval/return_mean",
        "eval/cost_mean",
        "eval/ep_len_mean",
        "eval/violation_rate_mean",
        "eval/crash_rate",
        "eval/timeout_rate",
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


def train(args, env_sampler, agent, pool, start_epoch=0):
    total_step = 0
    if start_epoch == 0:
        exploration_before_start(args, env_sampler, pool, agent)
    epoch = start_epoch
    train_metric_sums = {}
    train_metric_count = 0

    remaining_epochs = max(args.num_epoch - start_epoch, 0)
    for _ in range(remaining_epochs):
        sta = time.time()
        epo_len = args.epoch_length
        train_policy_steps = 0
        for i in range(epo_len):
            cur_state, action, next_state, reward, done, info = env_sampler.sample(agent, i)
            pool.push(cur_state, action, reward, next_state, env_sampler.last_terminated)

            # train the policy
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

            def evaluate(num_eval_epo=1):
                env_sampler.current_state = None
                avg_return, avg_cost_return = 0, 0
                eval_step = 0
                for _ in range(num_eval_epo):
                    sum_reward, sum_cost = 0, 0
                    eval_step = 0
                    done = False
                    while not done and eval_step < epo_len:
                        _, _, _, reward, done, _ = env_sampler.sample(agent, eval_step, eval_t=True)
                        sum_reward += reward[0]
                        sum_cost += reward[1] if args.safetygym else args.gamma**eval_step * reward[1]
                        eval_step += 1
                    avg_return += sum_reward
                    avg_cost_return += sum_cost
                avg_return /= num_eval_epo
                avg_cost_return /= num_eval_epo
                return avg_return, avg_cost_return

            if total_step % epo_len == 0 or total_step == 1:
                train_summary = {}
                if train_metric_count:
                    train_summary = {
                        key: value / train_metric_count
                        for key, value in train_metric_sums.items()
                    }

                if args.env_name == "Quad2D":
                    metrics = evaluate_quad2d_four_starts(env_sampler.env, agent, seed=args.seed)
                    env_sampler.current_state = None
                    env_sampler.needs_reset = True
                    env_sampler.path_length = 0

                    print(
                        f"env={args.env_name} step={total_step} "
                        f"return={metrics['eval/return_mean']:.2f} "
                        f"cost={metrics['eval/cost_mean']:.2f} "
                        f"viol={metrics['eval/violation_rate_mean']:.3f} "
                        f"len={metrics['eval/ep_len_mean']:.1f} "
                        f"crash={metrics['eval/crash_rate']:.2f} "
                        f"timeout={metrics['eval/timeout_rate']:.2f} "
                        f"q_loss={train_summary.get('train/critic_loss', float('nan')):.3g} "
                        f"|tq|={train_summary.get('train/target_q_abs_mean', float('nan')):.3g} "
                        f"qc_loss={train_summary.get('train/safety_critic_loss', float('nan')):.3g} "
                        f"lam={train_summary.get('train/lam', float('nan')):.3g} "
                        f"budget={args.cost_lim} "
                        f"time={int(time.time() - sta)}s"
                    )

                    if args.use_wandb:
                        log_dict = select_quad2d_wandb_metrics(metrics, train_summary)
                        log_dict["budget/cost_lim"] = args.cost_lim
                        log_dict["total_step"] = total_step
                        wandb.log(log_dict, step=total_step)

                else:
                    test_reward, test_cost = evaluate(args.num_eval_epochs)
                    print(
                        'env: {}, exp: {}, step: {}, test_return: {}, test_cost: {}, budget: {}, seed: {}, cuda_num: {}, time: {}s'.format(
                            args.env_name, args.experiment_name, total_step,
                            np.around(test_reward, 2), np.around(test_cost, 2),
                            args.cost_lim, args.seed, args.cuda_num, int(time.time() - sta)
                        )
                    )
                    if args.use_wandb:
                        wandb.log({"test_return": test_reward, "total_step": total_step})
                        wandb.log({"test_cost": test_cost, "total_step": total_step})

                train_metric_sums = {}
                train_metric_count = 0
        epoch += 1

        if args.save_parameters and epoch % 50 == 0:
            agent.save_model(suffix=f"{args.seed}_epoch{epoch}")
            mean, var, count = env_sampler.env.get_obs_stats()
            np.savez(
                f"models/{args.env_name}_{args.experiment_name}/obs_stats_{args.seed}_epoch{epoch}.npz",
                mean=mean, var=var, count=np.array(count)
            )

    # end of training
    if args.save_parameters:
        agent.save_model(suffix=f"{args.seed}_final")
        mean, var, count = env_sampler.env.get_obs_stats()
        np.savez(
            f"models/{args.env_name}_{args.experiment_name}/obs_stats_{args.seed}_final.npz",
            mean=mean, var=var, count=np.array(count)
        )


def exploration_before_start(args, env_sampler, pool, agent):
    for i in range(args.init_exploration_steps):
        cur_state, action, next_state, reward, done, info = env_sampler.sample(agent, i)
        pool.push(cur_state, action, reward, next_state, done)


def train_policy_repeats(args, total_step, train_step, pool, agent):
    if total_step % args.train_every_n_steps > 0:
        return 0, {}

    if train_step > args.max_train_repeat_per_step * total_step:
        return 0, {}

    metric_sums = {}
    metric_count = 0
    for i in range(args.num_train_repeat):
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = pool.sample(args.policy_train_batch_size)
        batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
        terminal_batch = batch_done.astype(bool)
        batch_done = (~terminal_batch).astype(int)
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


def get_next_run_dir(env_name):
    """Find the next available run index under models/{env_name}_run{N}/"""
    i = 1
    while os.path.exists(f"models/{env_name}_run{i}"):
        i += 1
    return f"models/{env_name}_run{i}", f"run{i}"


def main(args):
    torch.set_num_threads(args.n_training_threads)

    # Auto-assign experiment name only if not resuming and not provided
    if args.resume_dir is None and (not args.experiment_name or args.experiment_name == "exp"):
        i = 1
        while os.path.exists(f"models/{args.env_name}_exp{i}"):
            i += 1
        args.experiment_name = f"exp{i}"
        print(f"Auto-assigned experiment_name: {args.experiment_name}")

    run_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "results" / args.env_name / args.experiment_name

    if args.env_name == "Quad2D":
        env = Quad2DEnv()
        args.epoch_length = env.max_episode_steps
    elif args.env_name == "Quad3D":
        env = Quad3DEnv()
    else:
        env = gym.make(args.env_name)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    s_dim = env.observation_space.shape[0]

    if args.env_name == 'Ant-v3':
        s_dim = int(27)
    elif args.env_name == 'Humanoid-v3':
        s_dim = int(45)

    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if args.use_wandb:
        run = wandb.init(config=args,
                         project='SafeRL',
                         entity=args.user_name,
                         notes=socket.gethostname(),
                         name=args.experiment_name + '_' + str(args.cuda_num) + '_' + str(args.seed),
                         group=args.env_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)

    setproctitle.setproctitle(str(args.env_name) + "-" + str(args.seed))

    # Initial agent
    agent = CALAgent(s_dim, env.action_space, args)

    # Initial pool for env
    pool = ReplayMemory(args.replay_size)

    # Sampler of environment
    if args.safetygym or args.env_name == "Quad2D":
        env_sampler = SafetygymEnvSampler(args, env)
    else:
        env_sampler = MuJoCoEnvSampler(args, env)

    # Resume from checkpoint if requested
    start_epoch = 0
    if args.resume_dir is not None:
        paths = resolve_resume_paths(args.resume_dir, args.resume_suffix)

        agent.load_model(
            actor_path=paths["actor"],
            critics_path=paths["critics"],
            safetycritics_path=paths["safetycritics"],
        )

        if os.path.exists(paths["obs_stats"]):
            stats = np.load(paths["obs_stats"])
            env.set_obs_stats(stats["mean"], stats["var"], int(stats["count"]))
            print(f"Loaded obs stats from {paths['obs_stats']}")
        else:
            print(f"Warning: no obs stats at {paths['obs_stats']}, normalizer starts fresh")

        start_epoch = parse_epoch_from_suffix(paths["suffix"])
        print(f"Resuming from epoch {start_epoch}")

    train(args, env_sampler, agent, pool, start_epoch=start_epoch)

    if args.use_wandb:
        run.finish()


if __name__ == '__main__':
    args = readParser()
    if 'Safe' in args.env_name or args.env_name in ["Quad2D", "Quad3D"]:  # safetygym
        args.constraint_type = 'safetygym'
        args.safetygym = True
        args.epoch_length = 400
        # args.epoch_length = 500 if args.env_name == "Quad3D" else 400
    args.cost_lim = get_threshold(args.env_name, constraint=args.constraint_type)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_num
    # args.seed = torch.randint(0, 10000, (1,)).item()
    args.seed = 4921
    main(args)
