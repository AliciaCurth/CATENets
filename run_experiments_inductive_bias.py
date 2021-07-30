"""
Author: Alicia Curth
File to run experiments from shell
"""
import argparse

from experiments.experiments_inductive_bias.experiments_AB import do_acic_simu_loops
from experiments.experiments_inductive_bias.experiments_CD import do_ihdp_experiments


def init_arg():
    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", default='A', type=str)
    parser.add_argument("--file_name", default='results', type=str)
    parser.add_argument("--n_exp", default=10, type=int)
    parser.add_argument("--n_0", default=2000, type=int)
    parser.add_argument("--models", default=None, type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = init_arg()
    if (args.setup == 'A') or (args.setup == 'B'):
        do_acic_simu_loops(n_exp=args.n_exp, file_name=args.file_name, setting=args.setup,
                           n_0=args.n_0, models=args.models)
    elif (args.setup == 'C') or (args.setup == 'D'):
        do_ihdp_experiments(file_name=args.file_name, n_exp=args.n_exp, setting=args.setup
                            )
    else:
        raise ValueError('Setup should be one of A, B, C, D. You passed {}'.format(args.setup))
