"""
File to run experiments for
"On Inductive Biases for Heterogeneous Treatment Effect Estimation" (Curth & vdS, NeurIPS21)
from shell
"""
# Author: Alicia Curth
import argparse
import sys
from typing import Any

import catenets.logger as log
from experiments.experiments_inductivebias_NeurIPS21.experiments_AB import (
    do_acic_simu_loops,
)
from experiments.experiments_inductivebias_NeurIPS21.experiments_acic import (
    do_acic_orig_loop,
)
from experiments.experiments_inductivebias_NeurIPS21.experiments_CD import (
    do_ihdp_experiments,
)
from experiments.experiments_inductivebias_NeurIPS21.experiments_twins import (
    do_twins_experiment_loop,
)

log.add(sink=sys.stderr, level="DEBUG")


def init_arg() -> Any:
    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", default="A", type=str)
    parser.add_argument("--file_name", default="results", type=str)
    parser.add_argument("--n_exp", default=10, type=int)
    parser.add_argument("--n_0", default=2000, type=int)
    parser.add_argument("--models", default=None, type=str)
    parser.add_argument("--n1_loop", nargs="+", default=[200, 2000, 500], type=int)
    parser.add_argument(
        "--rho_loop", nargs="+", default=[0, 0.05, 0.1, 0.2, 0.5, 0.8], type=float
    )
    parser.add_argument("--factual_eval", default=False, type=bool)
    parser.add_argument(
        "--simu_nums", nargs="+", default=[x for x in range(1, 78)], type=int
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = init_arg()
    if (args.setup == "A") or (args.setup == "B"):
        do_acic_simu_loops(
            n_exp=args.n_exp,
            file_name=args.file_name,
            setting=args.setup,
            n_0=args.n_0,
            models=args.models,
            n1_loop=args.n1_loop,
            rho_loop=args.rho_loop,
            factual_eval=args.factual_eval,
        )
    elif (args.setup == "C") or (args.setup == "D"):
        do_ihdp_experiments(
            file_name=args.file_name, n_exp=args.n_exp, setting=args.setup
        )
    elif (args.setup == "acic") or (args.setup == "ACIC"):
        # Appendix E.1
        do_acic_orig_loop(
            simu_nums=args.simu_nums, n_exp=args.n_exp, file_name=args.file_name
        )
    elif (args.setup == "twins") or (args.setup == "Twins"):
        # Appendix E.2
        do_twins_experiment_loop(file_name=args.file_name, n_exp=args.n_exp)
    else:
        raise ValueError(
            f"Setup should be one of A, B, C, D, acic/ACIC or twins/Twins You passed"
            f" {args.setup}"
        )
