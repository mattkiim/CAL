import time
# import gym
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

EVAL_STARTS = [
    (1.0, 1.0),
    (-1.0, 1.0),
    (0.0, 0.53),
    (0.0, 1.47),
]


def evaluate_quad2d_four_starts(env, agent, seed=0):
    returns, costs, ep_lens, viol_rates = [], [], [], []

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

        while not done and steps < env.max_episode_steps:
            action = agent.select_action(obs, eval=True)  # adjust if your agent uses a different method
            obs, reward, cost, terminated, truncated, info = env.step(action)

            done = bool(terminated or truncated)
            ep_ret += float(reward)
            ep_cost += float(cost)
            steps += 1

        returns.append(ep_ret)
        costs.append(ep_cost)
        ep_lens.append(steps)
        viol_rates.append(ep_cost / float(env.max_episode_steps))

    returns = np.asarray(returns)
    costs = np.asarray(costs)
    ep_lens = np.asarray(ep_lens)
    viol_rates = np.asarray(viol_rates)

    return {
        "eval/return_mean": float(returns.mean()),
        "eval/return_std": float(returns.std()),
        "eval/cost_mean": float(costs.mean()),
        "eval/cost_std": float(costs.std()),
        "eval/ep_len_mean": float(ep_lens.mean()),
        "eval/ep_len_std": float(ep_lens.std()),
        "eval/violation_rate_mean": float(viol_rates.mean()),
        "eval/violation_rate_std": float(viol_rates.std()),
    }


def train(args, env_sampler, agent, pool):
    total_step = 0
    exploration_before_start(args, env_sampler, pool, agent)
    epoch = 0

    for _ in range(args.num_epoch):
        sta = time.time()
        epo_len = args.epoch_length
        train_policy_steps = 0
        for i in range(epo_len):
            cur_state, action, next_state, reward, done, info = env_sampler.sample(agent, i)
            pool.push(cur_state, action, reward, next_state, done)

            # train the policy
            if len(pool) > args.min_pool_size:
                train_policy_steps += train_policy_repeats(args, total_step, train_policy_steps, pool, agent)
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
                if args.env_name == "Quad2D":
                    metrics = evaluate_quad2d_four_starts(env_sampler.env, agent, seed=args.seed)

                    print(
                        f"env: {args.env_name}, step: {total_step}, "
                        f"return: {metrics['eval/return_mean']:.2f}, "
                        f"cost: {metrics['eval/cost_mean']:.2f}, "
                        f"viol_rate: {metrics['eval/violation_rate_mean']:.3f}, "
                        f"budget: {args.cost_lim}"
                    )

                    if args.use_wandb:
                        wandb.log(metrics, step=total_step)
                        wandb.log({"budget/cost_lim": args.cost_lim}, step=total_step)

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
        epoch += 1
        
        if args.save_parameters and epoch % 50 == 0:
            agent.save_model(suffix=f"{args.experiment_name}_{args.seed}_epoch{epoch}")
    
    # save network parameters after training
    if args.save_parameters:
        agent.save_model()


def exploration_before_start(args, env_sampler, pool, agent):
    for i in range(args.init_exploration_steps):
        cur_state, action, next_state, reward, done, info = env_sampler.sample(agent, i)
        pool.push(cur_state, action, reward, next_state, done)


def train_policy_repeats(args, total_step, train_step, pool, agent):
    if total_step % args.train_every_n_steps > 0:
        return 0

    if train_step > args.max_train_repeat_per_step * total_step:
        return 0

    for i in range(args.num_train_repeat):
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = pool.sample(args.policy_train_batch_size)
        batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
        batch_done = (~batch_done).astype(int)
        agent.update_parameters((batch_state, batch_action, batch_reward, batch_next_state, batch_done), i)
    return args.num_train_repeat


def main(args):
    torch.set_num_threads(args.n_training_threads)
    run_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "results" / args.env_name / args.experiment_name
    
    if args.env_name == "Quad2D":
        env = Quad2DEnv()
    else:
        env = gym.make(args.env_name)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # env.reset(seed=args.seed)

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
                         name= args.experiment_name + '_' + str(args.cuda_num) +'_' + str(args.seed),
                         group=args.env_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)

    setproctitle.setproctitle(str(args.env_name) + "-" + str(args.seed))

    # Intial agent
    agent = CALAgent(s_dim, env.action_space, args)

    # Initial pool for env
    pool = ReplayMemory(args.replay_size)

    # Sampler of environment
    if args.safetygym or args.env_name == "Quad2D":
        env_sampler = SafetygymEnvSampler(args, env)
    else:
        env_sampler = MuJoCoEnvSampler(args, env)

    train(args, env_sampler, agent, pool)

    if args.use_wandb:
        run.finish()


if __name__ == '__main__':
    args = readParser()
    if 'Safe' in args.env_name or args.env_name == "Quad2D": # safetygym
        args.constraint_type = 'safetygym'
        args.safetygym = True
        args.epoch_length = 400
    args.cost_lim = get_threshold(args.env_name, constraint=args.constraint_type)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_num
    args.seed = torch.randint(0, 10000, (1,)).item()
    main(args)