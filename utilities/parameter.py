import os
import argparse

#python main.py --train-flag --alpha 0.7 --pruned
#python main.py --train-flag --gate --ratio 0.7
def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu-flag', action='store_true',
                        help='flag for using gpu', default=True)
    parser.add_argument('--dynamic-pruning', action='store_true',
                        help='flag for dynamic_pruning', default=False)

    parser.add_argument('--train-flag', action='store_true',
                        help='flag for training network', default=False)

    parser.add_argument('--test-flag', action='store_true',
                        help='flag for testing network', default=False)

    parser.add_argument('--network', type=str,
                        help='Network for training', default='vgg')

    parser.add_argument('--data-set', type=str,
                        help='Data set for training network', default='CIFAR10')

    parser.add_argument('--data-path', type=str,
                        help='Path of dataset', default='./data')

    parser.add_argument('--epoch', type=int,
                        help='number of epoch for training network', default=10)
    #
    # parser.add_argument('--batch-size', type=int,
    #                     help='batch size', default=256)

    parser.add_argument('--lr', type=float,
                        help='learning rate', default=0.01)

    parser.add_argument('--lr-milestone', nargs='+', type=int,
                        help='list of epoch for adjust learning rate', default=None)

    parser.add_argument('--lr-gamma', type=float,
                        help='factor for decay learning rate', default=0.1)

    parser.add_argument('--momentum', type=float,
                        help='momentum for optimizer', default=0.9)

    parser.add_argument('--weight-decay', type=float,
                        help='factor for weight decay in optimizer', default=5e-4)

    parser.add_argument('--imsize', type=int,
                        help='size for image resize', default=32)

    parser.add_argument('--cropsize', type=int,
                        help='size for image crop', default=32)

    parser.add_argument('--crop-padding', type=int,
                        help='size for padding in image crop', default=4)

    parser.add_argument('--hflip', type=float,
                        help='probability of random horizontal flip', default=0.5)

    parser.add_argument('--load-path', type=str,
                        help='model load path', default=None)

    parser.add_argument('--save-path', type=str,
                        help='model save path', default="./checkpoint")

    parser.add_argument('--gated', action='store_true',
                        help='flag for gated', default=False)

    parser.add_argument('--ratio', type=float,
                        help='gated ratio', default=1)

    parser.add_argument('--pruned', action='store_true',
                        help='flag for filter pruned', default=False)

    parser.add_argument('--alpha', type=float,
                        help='prune rate', default=1)


    # parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))

    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    # parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox')

    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')
    # parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--model', type=str, default='vgg', help='neural network used in training')
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level', help='Different level of noise or different space of noise')
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--logdir', type=str, required=True, default="./logs/", help='Log directory path')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset used for training')
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    # parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=2,  help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='communication strategy: fedavg/fedprox')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/',
                        help='Checkpoint directory')
    parser.add_argument('--checkpoint_name', type=str, default='ckpt.pth.tar',
                        help='checkpoint path')
    return parser


def get_parameter():
    parser = build_parser()
    args = parser.parse_args()

    print("-*-" * 10 + "\n\t\tArguments\n" + "-*-" * 10)
    for key, value in vars(args).items():
        print("%s: %s" % (key, value))

    if args.save_path:
        save_folder = args.save_path[0:args.save_path.rindex('/')]
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            print("Make dir: ", save_folder)

    return args

