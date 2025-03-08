import torch
import botorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import random
from benchmark import get_problem
from LassoBO import run_LassoBO
from utils import save_args
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--func', default='levy15_300', type=str, )
parser.add_argument('--max_samples', default=1000, type=int)
parser.add_argument('--root_dir', default='synthetic_logs', type=str)
parser.add_argument('--dir_name', default=None, type=str)
parser.add_argument('--seed', default=2024, type=int)
args = parser.parse_args()

print(args)

random.seed(args.seed)
np.random.seed(args.seed)
botorch.manual_seed(args.seed)
torch.manual_seed(args.seed)

algo_name = 'LassoBO'
save_config = {
    'save_interval': 10,
    'root_dir': 'logs/' + args.root_dir,
    'algo': algo_name,
    'func': args.func if args.dir_name is None else args.dir_name,
    'seed': args.seed
}
f = get_problem(args.func, save_config, args.seed)

save_args(
    'config/' + args.root_dir,
    algo_name,
    args.func if args.dir_name is None else args.dir_name,
    args.seed,
    args
)
run_LassoBO(f, 30, args.max_samples)


