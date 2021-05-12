import argparse
from utils import *


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch RL trainer')
    # training
    # note: number of steps per epoch = epoch_size X batch_size x nprocesses
    parser.add_argument('--num_epochs', default=100, type=int,
                    help='number of training epochs')
    parser.add_argument('--epoch_size', type=int, default=50,
                    help='number of update iterations in an epoch')
    parser.add_argument('--batch_size', type=int, default=100,
                    help='number of steps before each update (per thread)')
    parser.add_argument('--nprocesses', type=int, default=1,
                    help='How many processes to run')
    # model
    parser.add_argument('--hid_size', default=64, type=int,
                    help='hidden layer size')
    parser.add_argument('--recurrent', action='store_true', default=False,
                    help='make the model recurrent in time')
    # optimization
    parser.add_argument('--gamma', type=float, default=0.6,
                    help='discount factor')
    parser.add_argument('--tau', type=float, default=1.0,
                    help='gae (remove?)')
    parser.add_argument('--seed', type=int, default=-1,
                    help='random seed. Pass -1 for random seed')  # TODO: works in thread?
    parser.add_argument('--normalize_rewards', action='store_true', default=False,
                    help='normalize rewards in each batch')
    parser.add_argument('--lrate', type=float, default=0.0005,
                    help='learning rate')
    parser.add_argument('--entr', type=float, default=0,
                    help='entropy regularization coeff')
    parser.add_argument('--value_coeff', type=float, default=0.01,
                    help='coeff for value loss term')
    # environment
    parser.add_argument('--env_name', default="Cartpole",
                    help='name of the environment to run')
    parser.add_argument('--max_steps', default=100, type=int,
                    help='force to end the game after this many steps')
    parser.add_argument('--nactions', default='1', type=str,
                    help='the number of agent actions (0 for continuous). Use N:M:K for multiple actions')
    parser.add_argument('--action_scale', default=1.0, type=float,
                    help='scale action output from model')
    # other
    parser.add_argument('--plot', action='store_true', default=False,
                    help='plot training progress')
    parser.add_argument('--plot_env', default='main', type=str,
                    help='plot env name')
    parser.add_argument('--save', default='', type=str,
                    help='save the model after training')
    parser.add_argument('--save_every', default=0, type=int,
                    help='save the model after every n_th epoch')
    parser.add_argument('--load', default='', type=str,
                    help='load the model')
    parser.add_argument('--display', action="store_true", default=False,
                    help='Display environment state')


    parser.add_argument('--random', action='store_true', default=False,
                    help="enable random model")

    # CommNet specific args
    parser.add_argument('--commnet', action='store_true', default=False,
                    help="enable commnet model")
    parser.add_argument('--ic3net', action='store_true', default=False,
                    help="enable commnet model")
    parser.add_argument('--comm_mode', type=str, default='avg',
                    help="Type of mode for communication tensor calculation [avg|sum]")
    parser.add_argument('--comm_passes', type=int, default=1,
                    help="Number of comm passes per step over the model")
    parser.add_argument('--comm_mask_zero', action='store_true', default=False,
                    help="Whether communication should be there")
    parser.add_argument('--mean_ratio', default=1.0, type=float,
                    help='how much coooperative to do? 1.0 means fully cooperative')
    parser.add_argument('--rnn_type', default='MLP', type=str,
                    help='type of rnn to use. [LSTM|MLP]')
    parser.add_argument('--detach_gap', default=10000, type=int,
                    help='detach hidden state and cell state for rnns at this interval.'
                    + ' Default 10000 (very high)')
    parser.add_argument('--comm_init', default='uniform', type=str,
                    help='how to initialise comm weights [uniform|zeros]')
    parser.add_argument('--hard_attn', default=False, action='store_true',
                    help='Whether to use hard attention: action - talk|silent')
    parser.add_argument('--comm_action_one', default=False, action='store_true',
                    help='Whether to always talk, sanity check for hard attention.')
    parser.add_argument('--advantages_per_action', default=False, action='store_true',
                    help='Whether to multipy log porb for each chosen action with advantages')
    parser.add_argument('--share_weights', default=False, action='store_true',
                    help='Share weights for hops')
    
    parser.add_argument('--record_video', action='store_true', default=False)
    parser.add_argument('--video_name', default='', type=str)
    parser.add_argument('--landmark_size', default=0.1, type=float)
    parser.add_argument('--num_landmarks',default=3,type=int)
    parser.add_argument('--arena_size', default=1.0, type=float)
    parser.add_argument('--collaborative',  action='store_true', default=False)
    parser.add_argument('--silent', default=True, type=bool)
    #simple_spread
    parser.add_argument('--nagents',default=3,type=int)
    parser.add_argument('--agent_size', default=0.2, type=float)

    #simple_tag
    parser.add_argument('--num_good_agents', default=1, type=int)
    parser.add_argument('--num_adversaries', default=2, type=int)
    parser.add_argument('--adversary_size', default=0.2, type=float)
    parser.add_argument('--good_size', default=0.1, type=float)
    parser.add_argument('--adversary_accel', default=3.0, type=float)
    parser.add_argument('--good_accel', default=1.0, type=float)
    parser.add_argument('--adversary_max_speed', default=1.0, type=float)
    parser.add_argument('--good_max_speed', default=0.5, type=float)

    parser.add_argument('--benchmark', default=False, action='store_true')
    parser.add_argument('--difficulty_level_threshold', default=20.0, type=float)

    init_args_for_env(parser)
    args = parser.parse_args()
    if args.env_name == 'simple_tag':
        args.nagents = args.num_good_agents + args.num_adversaries

    if args.video_name != '':
        args.record_video = True

    if args.record_video == True:
        args.display = True

    args.difficulty_level_threshold = args.max_steps/2
    return args
