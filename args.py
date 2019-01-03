import argparse


def get_args():
    parser = argparse.ArgumentParser()
#   args for the network
    parser.add_argument('--scale', default=2)

    parser.add_argument('--n_resblocks', default=3,
                        help='Number of Residual Blocks')

    parser.add_argument('--n_feats', default=64,
                        help='Feature maps in the Residual Block mapping path way' +
                        'The total Parameter of the network is:' +
                        '(x*x*6*1 + 6*x*x*0.8 + 3*3*x*x*0.8)*5 + 27*x + 25*12*3')

    parser.add_argument('--r_mean', type=float, default=0.4488,
                        help='Mean of R Channel')
    parser.add_argument('--g_mean', type=float, default=0.4371,
                        help='Mean of G channel')
    parser.add_argument('--b_mean', type=float, default=0.4040,
                        help='Mean of B channel')

    parser.add_argument('--n_colors', default=3)  # learning rate decrease every 5 epochs
    parser.add_argument('--res_scale', default=1)

#   args for the training
    parser.add_argument("--step", type=int, default=120,
                        help="Sets the learning rate to the initial LR decayed by 0.5 every 200 epoch")

    parser.add_argument("--start-epoch", default=1, type=int,
                        help="manual epoch number (useful on restarts)")
    parser.add_argument("--nEpochs", type=int, default=256,
                        help="number of epochs to train for")

    parser.add_argument('--bs', default=16,
                        help='Batch size')

    parser.add_argument("--cuda", action="store_true", default=False,
                        help="python py --cuda ==> True // python py ==> False")

    parser.add_argument("--resume", default='./checkpoint/model_epoch23_step1.pth', type=str,
                        help="path to latest checkpoint (default: none)")

#   args for the Optimizer
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')

    args = parser.parse_args()

    return args
