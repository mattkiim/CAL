import argparse

def readParser():
    parser = argparse.ArgumentParser(description='CAL')

    # ----------------------Env Config------------------------
    parser.add_argument('--env_name', default='Hopper-v3')
    # MuJoCo: 'Hopper-v3' 'HalfCheetah-v3' 'Ant-v3', 'Humanoid-v3'
    # Safety-Gym: 'Safexp-PointButton1-v0' 'Safexp-CarButton1-v0'
    # 'Safexp-PointButton2-v0' 'Safexp-CarButton2-v0' 'Safexp-PointPush1-v0'
    parser.add_argument('--safetygym', action='store_true', default=False)
    parser.add_argument('--constraint_type', default='velocity', help="['safetygym', 'velocity']")
    parser.add_argument('--epoch_length', type=int, default=1000) 
    parser.add_argument('--seed', type=int, default=123456)

    # -------------------Experiment Config---------------------
    parser.add_argument('--cuda', default=True, action="store_true",
                        help='run on CUDA (default: True)')
    parser.add_argument('--cuda_num', default='0', 
                        help='select the cuda number you want your program to run on')
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--user_name', default='')
    parser.add_argument('--n_training_threads', default=10)
    parser.add_argument('--experiment_name', default='exp')
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--num_eval_epochs', type=int, default=1)
    parser.add_argument('--save_parameters', action='store_true', default=False)

    # ---------------------Algorithm Config-------------------------
    parser.add_argument('--k', type=float, default=0.5)
    parser.add_argument('--qc_ens_size', type=int, default=4)
    parser.add_argument('--c', type=float, default=10)
    parser.add_argument('--num_train_repeat', type=int, default=10)

    parser.add_argument('--intrgt_max', action='store_true', default=False)
    parser.add_argument('--M', type=int, default=4, help='this number should be <= qc_ens_size')

    # -------------------Basic Hyperparameters---------------------
    parser.add_argument('--epsilon', default=1e-3)
    parser.add_argument('--init_exploration_steps', type=int, default=5000)
    parser.add_argument('--train_every_n_steps', type=int, default=1)
    parser.add_argument('--safety_gamma', type=float, default=0.99)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--qc_lr', type=float, default=0.0003)
    parser.add_argument('--critic_target_update_frequency', type=int, default=2)
    parser.add_argument('--replay_size', type=int, default=1000000)
    parser.add_argument('--min_pool_size', type=int, default=1000)
    parser.add_argument('--max_train_repeat_per_step', type=int, default=5)
    parser.add_argument('--policy_train_batch_size', type=int, default=12)
    parser.add_argument('--grad_clip_norm', type=float, default=10.0,
                        help='Global grad norm clip for actor/critic updates. <=0 disables clipping.')
    # ----- run more ----- #
    parser.add_argument("--resume_dir", default=None,
                        help="Directory to resume from, e.g. models/Quad2D_exp1")
    parser.add_argument("--resume_suffix", default=None,
                        help="Checkpoint suffix to resume from, e.g. 4921_epoch150. If omitted, picks latest.")

    # ---------------------Quad3D Training Config--------------------
    parser.add_argument("--quad3d_eval_interval", type=int, default=None,
                        help="Evaluate Quad3D every N environment steps. Defaults to 2500 in main_quad3d.py.")
    parser.add_argument("--quad3d_init_action_std", type=float, default=None,
                        help="Stddev for near-hover Quad3D warmup actions. Defaults to 0.05 in main_quad3d.py.")
    parser.add_argument("--quad3d_debug_eval", action="store_true", default=False,
                        help="Print per-start Quad3D eval diagnostics.")
    parser.add_argument("--quad3d_batch_size", type=int, default=None,
                        help="Quad3D-specific policy batch size. Defaults to 256 when the generic default is unchanged.")
    parser.add_argument("--quad3d_num_train_repeat", type=int, default=None,
                        help="Quad3D-specific gradient updates per env step. Defaults to 2 when the generic default is unchanged.")
    parser.add_argument("--quad3d_min_pool_size", type=int, default=None,
                        help="Quad3D-specific minimum replay size before training. Defaults to 5000 when the generic default is unchanged.")

    # ---------------------F16 Training Config--------------------
    parser.add_argument("--f16_eval_interval", type=int, default=None,
                        help="Evaluate F16 every N environment steps. Defaults to 500 in main_f16.py.")
    parser.add_argument("--f16_init_action_std", type=float, default=None,
                        help="Stddev for near-trim F16 warmup actions. Defaults to 0.05 in main_f16.py.")
    parser.add_argument("--f16_batch_size", type=int, default=None,
                        help="F16-specific policy batch size. Defaults to 256 when the generic default is unchanged.")
    parser.add_argument("--f16_num_train_repeat", type=int, default=None,
                        help="F16-specific gradient updates per env step. Defaults to 2 when the generic default is unchanged.")
    parser.add_argument("--f16_min_pool_size", type=int, default=None,
                        help="F16-specific minimum replay size before training. Defaults to 5000 when the generic default is unchanged.")

    return parser.parse_args()
