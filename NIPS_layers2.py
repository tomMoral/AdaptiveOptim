import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

if osp.exists(osp.join('/proc', 'acpi', 'bbswitch')):
    import os
    assert 'BUMBLEBEE_SOCKET' in os.environ.keys()

from Lcod.simple_problem_generator import SimpleProblemGenerator, \
    create_dictionary
from Lcod.lista_network import LIstaNetwork
from Lcod.lfista_network import LFistaNetwork
from Lcod.facto_network import FactoNetwork
from Lcod.ista_tf import IstaTF
from Lcod.fista_tf import FistaTF


def mk_curve(curve_cost, layer_lvl, eps=1e-6):
    # Plot the layer curve
    c_star = min(curve_cost['ista'][-1], curve_cost['fista'][-1])-eps

    fig = plt.figure('Curve layer')
    fig.clear()

    for model, name in [('lista', 'L-ISTA'), ('lfista', 'L-FISTA'),
                        ('facto', 'FacNet')]:
        cc = np.maximum(curve_cost[model]-c_star, eps)
        plt.loglog(layer_lvl, cc, 'o-', label=name)

    for model, name in [('ista', 'ISTA'), ('fista', 'FISTA')]:
        cc = np.maximum(curve_cost[model]-c_star, eps)
        t = range(1, len(cc))
        plt.loglog(t, cc[1:], label=name)

    plt.hlines([eps], 1, 600, 'r', '--', label='$E(z^*)$')
    # plt.legend(fontsize='xx-large')
    plt.xlim((1, 600))
    plt.xlabel('# iteration/layers', fontsize='x-large')
    plt.ylabel('Cost function', fontsize='x-large')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('Experiments for Adaopt paper')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='If set, the experiments will be saved in the '
                             'specified directory')
    parser.add_argument('--gpu', type=float, default=1.,
                        help='Ratio of usage of the gpu for this launch')
    parser.add_argument('--lmbd', type=float, default=.01,
                        help='Lambda used for the experiments, control the '
                             'regularisation level')
    parser.add_argument('--rho', type=float, default=.05,
                        help='Rho used for the experiments, control the '
                             'sparsity level')
    parser.add_argument('--debug', '-d', type=int, default=20,
                        help='Logging level, default is INFO, '
                             'set 10 for DEBUG')
    args = parser.parse_args()

    # General Setup and constants
    # Experiment metadata
    save_exp = args.save_dir is not None
    NAME_EXP = args.save_dir if args.save_dir else 'default'
    gpu_usage = args.gpu  # GPU memory allocated to this process
    log_lvl = args.debug  # verbosity -- INFO

    # Constants
    K = 100                # Number of dictionary atoms
    p = 64                 # Dimension of the data
    N_test = 10000         # Number of test sample
    N_val = 1000           # Number of test sample

    lmbd = args.lmbd        # Regularisation level
    rho = args.rho          # Sparsity level
    corr = 0                # Correlation level for the coefficients
    eps = 1e-6              # Resolution for the optimization problem

    # Extra network params
    warm_params = True
    lr_init = 1e-1
    steps = 100
    batch_size = 500

    # Setup the experiment plan
    it_lista = 300
    it_lfista = 300
    it_facto = 300
    run_exps = [{'n_layers': 1,
                 'lista': it_lista, 'lfista': 0, 'facto': it_facto},
                {'n_layers': 2,
                 'lista': it_lista, 'lfista': it_lfista, 'facto': it_facto},
                {'n_layers': 4,
                 'lista': it_lista, 'lfista': it_lfista, 'facto': it_facto},
                {'n_layers': 7,
                 'lista': it_lista, 'lfista': it_lfista, 'facto': it_facto},
                {'n_layers': 12,
                 'lista': it_lista, 'lfista': it_lfista, 'facto': it_facto},
                {'n_layers': 21,
                 'lista': it_lista, 'lfista': it_lfista, 'facto': it_facto},
                {'n_layers': 35,
                 'lista': it_lista, 'lfista': it_lfista, 'facto': it_facto},
                {'n_layers': 59,
                 'lista': it_lista, 'lfista': it_lfista, 'facto': it_facto},
                {'n_layers': 100,
                 'lista': it_lista, 'lfista': it_lfista, 'facto': it_facto},
                ]
    layer_lvl = [v['n_layers'] for v in run_exps]

    # Setup the training constant and a test set
    D = create_dictionary(K, p, seed=290890)
    pb = SimpleProblemGenerator(D, lmbd, rho=rho, batch_size=batch_size,
                                corr=corr, seed=422742)

    sig_test, z0_test, zs_test, _ = pb.get_batch(N_test)
    sig_val, z0_val, zs_val, _ = pb.get_batch(N_val)

    # Compute optimal values for validation/test sets
    ista = IstaTF(D, gpu_usage=gpu_usage)
    zs_t = np.copy(zs_test)
    ista.optimize(X=sig_val, lmbd=lmbd, Z=zs_val,
                  max_iter=5000, tol=1e-6)
    c_val = ista.train_cost[-1]
    ista.optimize(X=sig_test, lmbd=lmbd, Z=zs_t,
                  max_iter=5000, tol=1e-6)
    fista = FistaTF(D, gpu_usage=gpu_usage)
    zs_t = np.copy(zs_test)
    fista.optimize(X=sig_val, lmbd=lmbd, Z=zs_val,
                   max_iter=5000, tol=1e-6)
    c_val = min(c_val, fista.train_cost[-1])
    fista.optimize(X=sig_test, lmbd=lmbd, Z=zs_t,
                   max_iter=5000, tol=1e-6)

    feed_test = {"Z": zs_test, "X": sig_test, "lmbd": lmbd}
    feed_val = {"Z": zs_val, "X": sig_val, "lmbd": lmbd, "c_val": c_val-1e-10}

    # Free the ressources
    ista.terminate()
    fista.terminate()

    # Setup saving variables
    SAVE_DIR = osp.join('save_exp', NAME_EXP)
    if not osp.exists(SAVE_DIR):
        import os
        os.mkdir(SAVE_DIR)
        os.mkdir(osp.join(SAVE_DIR, 'trn_cost'))

    # Reload past experiment points
    networks = {}
    try:
        curve_cost = np.load(osp.join(SAVE_DIR, "curve_cost.npy")).take(0)
    except FileNotFoundError:
        c_star = min(ista.train_cost[-1], fista.train_cost[-1])-eps
        curve_cost = {'lista': (2+c_star)*np.ones(len(layer_lvl)),
                      'lfista': (2+c_star)*np.ones(len(layer_lvl)),
                      'facto': (2+c_star)*np.ones(len(layer_lvl)),
                      'ista': ista.train_cost,
                      'fista': fista.train_cost
                      }

    # Run the experiments
    models = [('lista', LIstaNetwork), ('lfista', LFistaNetwork),
              ('facto', FactoNetwork)]
    wp = {}
    for m, _ in models:
        wp[m] = []

    for i, expe in enumerate(run_exps):
        n_layers = expe['n_layers']
        for model, obj in models:
            key = '{}_{}'.format(model, n_layers)
            if expe[model] > 0:
                if key not in networks.keys():
                    network = networks[key] = obj(
                        D, n_layers=n_layers, shared=False, log_lvl=log_lvl,
                        gpu_usage=gpu_usage, warm_param=wp[model],
                        exp_dir=NAME_EXP)
                else:
                    network = networks[key]
                    # network.reset()
                network.train(pb, expe[model], steps, feed_val,
                              lr_init=lr_init if 'facto' != model else 1e-4,
                              reg_cost=8, tol=1e-8,
                              model_name='layers/{}'.format(key))
                if warm_params:
                    wp[model] = network.export_param()

                curve_cost[model][i] = network.cost(**feed_test)
                np.save(osp.join(SAVE_DIR, 'trn_cost', key), network.cost_val)
                np.save(osp.join(SAVE_DIR, "curve_cost"), curve_cost)
                network.terminate()
            elif warm_params and key in networks.keys():
                wp[model] = networks[key].export_param()

    curve_cost['lfista'][0] = curve_cost['lista'][0]

    np.save(osp.join(SAVE_DIR, "curve_cost"), curve_cost)

    import IPython
    IPython.embed()

    if save_exp:
        save_value = dict(layer_lvl=layer_lvl, curve_cost=curve_cost, pb=pb,
                          L=L)
        import pickle
        from datetime import datetime
        t = datetime.now()
        save_file = 'save_layer{0.day:02}{0.month:02}.pkl'.format(t)
        with open(osp.join(SAVE_DIR, save_file), 'wb') as f:
            pickle.dump(save_value, f)
