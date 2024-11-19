import argparse
import random

from train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='test', help='experiment name')
    parser.add_argument('--env_name', type=str, default='coinrun', help='environment ID')
    parser.add_argument('--val_env_name', type=str, default=None, help='optional validation environment ID')
    parser.add_argument('--start_level', type=int, default=int(0), help='start-level for environment')
    parser.add_argument('--num_levels', type=int, default=int(0), help='number of training levels for environment')
    parser.add_argument('--distribution_mode', type=str, default='easy', help='distribution mode for environment')
    parser.add_argument('--param_name', type=str, default='easy-200', help='hyper-parameter ID')
    parser.add_argument('--device', type=str, default='gpu', required=False, help='whether to use gpu')
    parser.add_argument('--gpu_device', type=int, default=int(0), required=False, help='visible device in CUDA')
    parser.add_argument('--num_timesteps', type=int, default=int(25000000), help='number of training timesteps')
    parser.add_argument('--seed', type=int, default=random.randint(0, 9999), help='Random generator seed')
    parser.add_argument('--log_level', type=int, default=int(40), help='[10,20,30,40]')
    parser.add_argument('--num_checkpoints', type=int, default=int(1), help='number of checkpoints to store')
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--use_wandb', action="store_true")

    parser.add_argument('--wandb_tags', type=str, nargs='+')

    parser.add_argument('--random_percent', type=int, default=0,
                        help='COINRUN: percent of environments in which coin is randomized (only for coinrun)')
    parser.add_argument('--key_penalty', type=int, default=0,
                        help='HEIST_AISC: Penalty for picking up keys (divided by 10)')
    parser.add_argument('--step_penalty', type=int, default=0,
                        help='HEIST_AISC: Time penalty per step (divided by 1000)')
    parser.add_argument('--rand_region', type=int, default=0,
                        help='MAZE: size of region (in upper left corner) in which goal is sampled.')

    # multi threading
    parser.add_argument('--num_threads', type=int, default=8)
    args = parser.parse_args()

    args.exp_name ="ascent"
    args.env_name ="ascent"
    args.param_name ="ascent-misgen"
    args.num_checkpoints= 1
    args.seed = 1080
    args.use_wandb = True
    # args.model_file = "logs/train/ascent/Ascent/2024-11-18__11-17-05__seed_81/model_10027008.pth"
    args.n_pos_states = 10
    args.epoch = 0
    args.num_timesteps = int(65000)
    args.val_epoch = 200
    train(args)

