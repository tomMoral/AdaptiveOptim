try:
    import sys
    sys.path.remove("/usr/lib/python3/dist-packages")
except ValueError:
    pass

import os
import json
import numpy as np
import matplotlib.pyplot as plt

if os.path.exists(os.path.join('/proc', 'acpi', 'bbswitch')):
    # assert that the graphic card is on if bbswitch is detected
    import os
    assert 'BUMBLEBEE_SOCKET' in os.environ.keys()

from Lcod.fista_tf import FistaTF
from Lcod.facto_network import FactoNetwork
from Lcod.linear_network import LinearNetwork


def get_problem(config):

    # retrieve the parameter of the problem
    dataset = config['data']
    batch_size, lmbd = config['batch_size'], config['lmbd']
    seed = config.get('seed')

    # Setup the training constant and a test set
    if dataset == 'artificial':
        from Lcod.simple_problem_generator import SimpleProblemGenerator
        from Lcod.simple_problem_generator import create_dictionary

        # retrieve specific parameters for the problem
        K, p, rho = config['K'], config['p'], config['rho']
        seed_D, corr = config.get('seed_D'), config.get('corr', 0)
        D = create_dictionary(K, p, seed=seed_D)
        pb = SimpleProblemGenerator(D, lmbd, rho=rho, batch_size=batch_size,
                                    corr=corr, seed=seed)
    elif dataset == 'adverse':
        from Lcod.simple_problem_generator import SimpleProblemGenerator
        from data_handlers.dictionaries import create_adversarial_dictionary

        # retrieve specific parameters for the problem
        K, p, rho = config['K'], config['p'], config['rho']
        seed_D, corr = config.get('seed_D'), config.get('corr', 0)
        D = create_adversarial_dictionary(K, p, seed=seed_D)
        pb = SimpleProblemGenerator(D, lmbd, rho=rho, batch_size=batch_size,
                                    corr=corr, seed=seed)
    elif dataset == 'mnist':
        from Lcod.mnist_problem_generator import MnistProblemGenerator
        from Lcod.mnist_problem_generator import create_dictionary_dl
        K, save_dir = config['K'], config['save_dir']
        D = create_dictionary_dl(lmbd, K, N=10000, dir_mnist=save_dir)
        pb = MnistProblemGenerator(D, lmbd, batch_size=batch_size,
                                   dir_mnist=save_dir, seed=seed)
    elif dataset == 'images':
        from Lcod.image_problem_generator import ImageProblemGenerator
        from Lcod.image_problem_generator import create_dictionary_haar
        p = config['p']
        D = create_dictionary_haar(p)
        pb = ImageProblemGenerator(D, lmbd, batch_size=batch_size,
                                   seed=seed)
    else:
        raise NameError("dataset {} not reconized by the script"
                        "".format(dataset))
    return pb, D


def _assert_exist(*args):
    """create a directory if it does not exist."""
    path = os.path.join(*args)
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def parse_runfile(file):
    with open(file) as f:
        runs = json.load(f)
    run_exps = runs['run_exps']
    for exp in run_exps:
        for k, v in exp.items():
            if type(v) is str:
                exp[k] = runs[v]
    return run_exps


def load_exp(exp_name):
    exp_dir = os.path.join("exps", exp_name)
    file = os.path.join(exp_dir, "config.json")
    with open(file) as f:
        exps = json.load(f)
    exps['save_dir'] = exp_dir
    run_exps = parse_runfile(os.path.join(exp_dir, exps["runfile"]))
    pb, D = get_problem(exps)
    return D, pb, run_exps, exps


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('Experiments for Adaopt paper')
    parser.add_argument('--exp', type=str, default="default",
                        help='If set, the experiments will be saved in the '
                             'specified directory')
    parser.add_argument('--gpu', type=float, default=.95,
                        help='Ratio of usage of the gpu for this launch')
    parser.add_argument('--debug', '-d', type=int, default=20,
                        help='Logging level, default is INFO, '
                             'set 10 for DEBUG')
    parser.add_argument('--linear', '-l', action='store_true',
                        help='recompute the linear layer model')
    args = parser.parse_args()

    # General Setup and constants
    # Experiment metadata
    exp_name = args.exp   # name of the experiment
    gpu_usage = args.gpu  # GPU memory allocated to this process
    log_lvl = args.debug  # verbosity -- INFO

    # Setup saving variables
    _assert_exist('exps')
    save_dir = _assert_exist('exps', exp_name)
    _assert_exist(save_dir, 'ckpt')
    save_curve = os.path.join(save_dir, "curve_cost.npy")

    D, pb, run_exps, exps = load_exp(exp_name)

    run_exps = [
        # {
        #     'name': 'manifold.1_sg',
        #     'manifold': True,
        #     'proj_A': True,
        #     'reg_unary': False,
        #     'run_svd': False,
        #     'reg_scale': 0,
        #     'sgd': True,
        #     'lr_init': .00001
        # },
        # {
        #     'name': 'manifold_reg.1_rs-4_sg',
        #     'manifold': True,
        #     'proj_A': True,
        #     'reg_unary': True,
        #     'run_svd': False,
        #     'reg_scale': 1e-4,
        #     'sgd': True,
        #     'lr_init': .00001
        # },
        # {
        #     'name': 'reg.1_rs-4_f',
        #     'manifold': False,
        #     'proj_A': False,
        #     'reg_unary': True,
        #     'run_svd': True,
        #     'reg_scale': 1e-4,
        #     'sgd': False,
        #     'lr_init': .1
        # },
        # { # Best for now
        #     'name': 'reg_proj.1_rs-4_k',
        #     'manifold': False,
        #     'proj_A': True,
        #     'reg_unary': True,
        #     'run_svd': True,
        #     'reg_scale': 1e-4,
        #     'sgd': False,
        #     'init_value_ada': 1e-2,
        #     'lr_init': .05
        # }
        # {
        #     'name': 'reg_proj.1_rs-4_l3',
        #     'manifold': False,
        #     'proj_A': True,
        #     'reg_unary': True,
        #     'run_svd': True,
        #     'reg_scale': 1e-4,
        #     'sgd': False,
        #     'init_value_ada': 5e-2,
        #     'lr_init': .1
        # },
        {
            'name': 'reg_proj.1_rs-4_m2',
            'manifold': False,
            'proj_A': False,
            'reg_unary': True,
            'run_svd': False,
            'reg_scale': 1e4,
            'sgd': False,
            'init_value_ada': 1e-2,
            'lr_init': .1
        }
    ]

    # Load variable value from json
    lmbd, lr_init, reg_scale, steps, warm_params = [
        exps.get(k) for k in ['lmbd', 'lr_init', 'reg_scale',
                              'steps', 'warm_params']]

    sig_test, z0_test, zs_test, _ = pb.get_test(exps['N_test'])
    sig_val, z0_val, zs_val, _ = pb.get_batch(exps['N_val'])
    C0 = pb.lasso_cost(zs_test, sig_test)

    # Compute the validation cost
    fista = FistaTF(D, gpu_usage=gpu_usage)
    fista.optimize(X=sig_val, lmbd=lmbd, Z=zs_val,
                   max_iter=10000, tol=1e-8*C0)
    fista.terminate()
    c_val = fista.train_cost[-1]

    feed_test = {"Z": zs_test, "X": sig_test, "lmbd": lmbd}
    feed_val = {"Z": zs_val, "X": sig_val, "lmbd": lmbd, "c_val": c_val-1e-10}

    # Reload past experiment points
    networks = {}
    curve_cost = np.load(save_curve).take(0)

    lin_net = LinearNetwork(D, 1, exp_dir=exp_name)
    lin_net.saver.restore(lin_net.session,
                          "exps/mnist_100_05/ckpt/linear-50000")
    print("Cost:", lin_net.cost(**feed_test)-curve_cost['fista'][-1])

    # Run the experiments
    models = [('facto', FactoNetwork)]
    wp = {}
    for m, _ in models:
        wp[m] = []

    c_star = min(min(curve_cost['ista']), min(curve_cost['fista']))
    for i, expe in enumerate(run_exps):
        lr_init = expe.pop("lr_init")
        network = FactoNetwork(
            D, n_layers=1, shared=False, log_lvl=log_lvl,
            gpu_usage=gpu_usage, exp_dir="test_fact", **expe)
        network.train(
            pb, feed_val, max_iter=500, steps=steps, reg_cost=8, tol=1e-8,
            # lr_init=lr_init if 'facto' != model else lr_fn/n_layers,
            lr_init=lr_init)
    import IPython
    IPython.embed()
