import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

if osp.exists(osp.join('/proc', 'acpi', 'bbswitch')):
    # assert that the graphic card is on if bbswitch is detected
    import os
    assert 'BUMBLEBEE_SOCKET' in os.environ.keys()

from Lcod.lista_network import LIstaNetwork
from Lcod.lfista_network import LFistaNetwork
from Lcod.facto_network import FactoNetwork
from Lcod.ista_tf import IstaTF
from Lcod.fista_tf import FistaTF


def mk_curve(curve_cost, max_iter=1000, eps=1e-6):
    # Plot the layer curve
    c_star = min(curve_cost['ista'][-1], curve_cost['fista'][-1])-eps

    fig = plt.figure('Curve layer')
    fig.clear()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(bottom=.12, top=.99)

    y_max = 0
    sym = 10
    max_iter = min(max_iter, len(curve_cost['ista']))
    layer_lvl = [1, 2, 4, 7, 12, 21, 35, 59, 100]

    for model, name, style in [('lista', 'L-ISTA', 'bo-'),
                               ('lfista', 'L-FISTA', 'c*-'),
                               ('facto', 'FacNet', 'rd-')]:
        cc = np.maximum(curve_cost[model]-c_star, eps)
        y_max = max(y_max, cc[0])
        ax.loglog(layer_lvl, cc, style, label=name)

    for model, name, style in [('ista', 'ISTA', 'g-'),
                               ('fista', 'FISTA', 'ys-')]:
        cc = np.maximum(curve_cost[model]-c_star, eps)
        y_max = max(y_max, cc[0])
        iters = min(max_iter, len(cc))
        makers = np.unique((10**np.arange(0, np.log10(iters-1), 2/9)
                            ).astype(int))-1
        t = range(1, len(cc))
        ax.loglog(t, cc[1:], style,
                  markevery=makers,
                  label=name)

    ax.hlines([eps], 1, max_iter, 'k', '--', label='$E(z^*)$')

    ax.legend(fontsize='x-large', ncol=2)
    ax.set_xlim((1, max_iter))
    ax.set_ylim((eps/2, sym*y_max))
    ax.set_xlabel('# iteration/layers k', fontsize='x-large')
    ax.set_ylabel('Cost function $F(z) - F(z^*)$', fontsize='x-large')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(10)


def get_problem(dataset, K, p, lmbd, rho, batch_size, save_dir):
    # Setup the training constant and a test set
    if dataset == 'artificial':
        from Lcod.simple_problem_generator import SimpleProblemGenerator
        from Lcod.simple_problem_generator import create_dictionary
        D = create_dictionary(K, p, seed=290890)
        pb = SimpleProblemGenerator(D, lmbd, rho=rho, batch_size=batch_size,
                                    corr=corr, seed=422742)
    elif dataset == 'mnist':
        from Lcod.mnist_problem_generator import MnistProblemGenerator
        from Lcod.mnist_problem_generator import create_dictionary_dl
        D = create_dictionary_dl(lmbd, K, N=10000, dir_mnist=save_dir)
        pb = MnistProblemGenerator(D, lmbd, batch_size=batch_size,
                                   dir_mnist=save_dir, seed=42242)
    elif dataset == 'images':
        from Lcod.image_problem_generator import ImageProblemGenerator
        from Lcod.image_problem_generator import create_dictionary_haar
        p = int(np.sqrt(p))
        D = create_dictionary_haar(p, wavelet='haar')
        pb = ImageProblemGenerator(D, lmbd, batch_size=batch_size,
                                   seed=1234)
    else:
        raise NameError("dataset {} not reconized by the script"
                        "".format(dataset))
    return pb, D


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
    parser.add_argument('--data', type=str, default='artificial',
                        help='Dataset to run the experiments. Can be one of '
                             '["artificial", "images", "mnist"]')
    parser.add_argument('-K', type=int, default=100,
                        help='Number of dictionary elements used.')
    args = parser.parse_args()

    # General Setup and constants
    # Experiment metadata
    save_exp = args.save_dir is not None
    NAME_EXP = args.save_dir if args.save_dir else 'default'
    gpu_usage = args.gpu  # GPU memory allocated to this process
    log_lvl = args.debug  # verbosity -- INFO
    dataset = args.data   # dataset to run the experiment

    # Constants
    K = args.K             # Number of dictionary atoms
    N_test = 10000         # Number of test sample
    N_val = 1000           # Number of test sample

    lmbd = args.lmbd        # Regularisation level
    rho = args.rho          # Sparsity level
    corr = 0                # Correlation level for the coefficients
    eps = 1e-6              # Resolution for the optimization problem
    reg_scale = 1           # scaling of the unitary penalization

    # Extra network params
    warm_params = True      # Reuse the parameters from smaller network
    lr_init = 5e-2          # Initial learning rate for the gradient descent
    lr_fn = 1e-3            # Initial learning rate for GD in FacNet
    steps = 100             # Number of steps fo GD between validation
    batch_size = 300        # Size of the batch for the training

    # Setup the experiment plan
    it_lista = it_lfista = it_facto = 500
    run_exps = [
        {'n_layers': 1,
         'lista': it_lista, 'lfista': it_lfista, 'facto': it_facto},
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

    # Setup saving variables
    SAVE_DIR = osp.join('save_exp', NAME_EXP)
    if not osp.exists(SAVE_DIR):
        import os
        os.mkdir(SAVE_DIR)
        os.mkdir(osp.join(SAVE_DIR, 'trn_cost'))

    # Setup the training constant and a test set
    if dataset == 'artificial':
        from Lcod.simple_problem_generator import SimpleProblemGenerator
        from Lcod.simple_problem_generator import create_dictionary
        p = 64                 # Dimension of the data
        D = create_dictionary(K, p, seed=290890)
        pb = SimpleProblemGenerator(D, lmbd, rho=rho, batch_size=batch_size,
                                    corr=corr, seed=422742)
    elif dataset == 'mnist':
        from Lcod.mnist_problem_generator import MnistProblemGenerator
        from Lcod.mnist_problem_generator import create_dictionary_dl
        D = create_dictionary_dl(lmbd, K, N=10000, dir_mnist=SAVE_DIR)
        pb = MnistProblemGenerator(D, lmbd, batch_size=batch_size,
                                   dir_mnist=SAVE_DIR, seed=42242)
    elif dataset == 'images':
        from Lcod.image_problem_generator import ImageProblemGenerator
        from Lcod.image_problem_generator import create_dictionary_haar
        p = 8
        reg_scale = 1e-4
        D = create_dictionary_haar(p)
        pb = ImageProblemGenerator(D, lmbd, batch_size=batch_size,
                                   data_dir='data/VOC', seed=1234)
    else:
        raise NameError("dataset {} not reconized by the script"
                        "".format(dataset))

    sig_test, z0_test, zs_test, _ = pb.get_test(N_test)
    sig_val, z0_val, zs_val, _ = pb.get_batch(N_val)
    C0 = pb.lasso_cost(zs_test, sig_test)

    # Compute optimal values for validation/test sets
    ista = IstaTF(D, gpu_usage=gpu_usage)
    ista.optimize(X=sig_val, lmbd=lmbd, Z=zs_val,
                  max_iter=10000, tol=1e-8*C0)
    c_val = ista.train_cost[-1]
    ista.optimize(X=sig_test, lmbd=lmbd, Z=zs_test,
                  max_iter=10000, tol=1e-8*C0)
    fista = FistaTF(D, gpu_usage=gpu_usage)
    fista.optimize(X=sig_val, lmbd=lmbd, Z=zs_val,
                   max_iter=10000, tol=1e-8*C0)
    c_val = min(c_val, fista.train_cost[-1])
    fista.optimize(X=sig_test, lmbd=lmbd, Z=zs_test,
                   max_iter=10000, tol=1e-8*C0)

    feed_test = {"Z": zs_test, "X": sig_test, "lmbd": lmbd}
    feed_val = {"Z": zs_val, "X": sig_val, "lmbd": lmbd, "c_val": c_val-1e-10}

    # Free the ressources
    ista.terminate()
    fista.terminate()

    c_star = min(ista.train_cost[-1],
                 fista.train_cost[-1])
    print(ista.train_cost[-1], fista.train_cost[-1], C0)

    # Reload past experiment points
    networks = {}
    try:
        curve_cost = np.load(osp.join(SAVE_DIR, "curve_cost.npy")).take(0)
        curve_cost['ista'] = ista.train_cost
        curve_cost['fista'] = fista.train_cost
    except FileNotFoundError:
        c_star = min(ista.train_cost[-1], fista.train_cost[-1])-eps
        curve_cost = {'lista': 2*C0*np.ones(len(layer_lvl)),
                      'lfista': 2*C0*np.ones(len(layer_lvl)),
                      'facto': 2*C0*np.ones(len(layer_lvl)),
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
                        exp_dir=NAME_EXP, reg_scale=reg_scale)
                else:
                    network = networks[key]
                    # network.reset()
                network.train(
                    pb, expe[model], steps, feed_val, reg_cost=8, tol=1e-8,
                    lr_init=lr_init if 'facto' != model else lr_fn/n_layers,
                    model_name='layers/{}'.format(key))
                if warm_params:
                    wp[model] = network.export_param()

                curve_cost[model][i] = network.cost(**feed_test)
                np.save(osp.join(SAVE_DIR, 'trn_cost', key), network.cost_val)
                np.save(osp.join(SAVE_DIR, "curve_cost"), curve_cost)
                try:
                    np.save(osp.join(
                        SAVE_DIR, 'trn_cost', '{}_weights'.format(key)),
                            [[]] + network.export_param())
                except ValueError:
                    print("Error in param saving for model {}".format(key))
                    raise
                network.terminate()
            elif warm_params and key in networks.keys():
                wp[model] = networks[key].export_param()
            elif warm_params:
                try:
                    wp[model] = np.load(osp.join(
                        SAVE_DIR, 'trn_cost', '{}_weights'.format(key)))[1:]
                except FileNotFoundError:
                    pass

    curve_cost['lfista'][0] = curve_cost['lista'][0]
    np.save(osp.join(SAVE_DIR, "curve_cost"), curve_cost)

    if save_exp:
        save_value = dict(layer_lvl=layer_lvl, curve_cost=curve_cost, pb=pb)
        import pickle
        from datetime import datetime
        t = datetime.now()
        save_file = 'save_layer{0.day:02}{0.month:02}.pkl'.format(t)
        with open(osp.join(SAVE_DIR, save_file), 'wb') as f:
            pickle.dump(save_value, f)

    import IPython
    IPython.embed()
