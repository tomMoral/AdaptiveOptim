# coding: utf-8
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

from Lcod.mnist_problem_generator import MnistProblemGenerator,\
    create_dictionary_dl
from Lcod.lista_network import LIstaNetwork
from Lcod.lfista_network import LFistaNetwork
from Lcod.ista_tf import IstaTF
from Lcod.fista_tf import FistaTF


def checkpoint(network, SAVE_DIR):
    np.save(osp.join(SAVE_DIR, 'train_cost/{}'.format(network.name)),
            network.train_cost)
    np.save(osp.join(SAVE_DIR, 'w_{}.npy'.format(network.name)),
            network.export_param())
    network.terminate()


if __name__ == '__main__':
    # General Setup and constants

    # Constants
    K = 239                # Number of dictionary atoms
    p = 8                  # Dimension of the data
    lmbd = .01             # Regularisation level
    N_test = 10000         # Number of test samples
    N_val = 1000           # Number of validation samples


    # Setup the training constant and a test set
    steps = 100
    batch_size = 500
    max_iter = 400

    # Setup saving variables
    save_exp = True
    NAME_EXP = 'mnist'
    SAVE_DIR = osp.join('save_exp', NAME_EXP)
    if not osp.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
        os.mkdir(osp.join(SAVE_DIR, 'train_cost'))

    # Setup optimization options
    warm_param = True   # Use the previously trained layers for later networks
    reload_net = False  # Load the previously stored models
    gpu_usage = 1       # Fraction of the GPU disponible for this script
    lr = 2e-1           # Initial learning rate
    max_iter = 1000     # Maximal number of training iteration

    # Experiement planing. (run ISTA, run FISTA)
    run_net_lvl = [(True, False), (True, True), (True, True), (True, True),
                   (True, True), (True, True), (True, True), (True, True),
                   (True, True)]

    layer_lvl = np.unique(np.logspace(0, 2, 10).astype(int))
    print("NB layers: {}".format(len(layer_lvl)))

    # Setup the lasso problem
    fname = osp.join(SAVE_DIR, "D_mnist_K{}_lmbd{}.npy".format(K, lmbd))
    if os.path.exists(fname):
        D = np.load(fname)
        D_100 = np.load("D_mnist_K100_lmbd{}.npy".format(lmbd))
    else:
        D = create_dictionary_dl(lmbd=lmbd, K=K, N=N_test//4)
        np.save(fname, D)

    pb = MnistProblemGenerator(D, lmbd, batch_size)

    # draw test and validation set and compute the optimal cost function
    sig_test, z0_test, zs_test, _ = pb.get_batch(N_test)
    sig_val, z0_val, zs_val, _ = pb.get_batch(N_val)

    ista = IstaTF(D)
    zs_t = np.copy(zs_test)
    ista.optimize(X=sig_val, lmbd=lmbd, Z=zs_val,
                  max_iter=2*max_iter, tol=1e-12)
    c_val = ista.train_cost[-1]
    ista.optimize(X=sig_test, lmbd=lmbd, Z=zs_t,
                  max_iter=2*max_iter, tol=1e-15)
    c_tst = ista.train_cost[-1]
    fista = FistaTF(D)
    zs_t = np.copy(zs_test)
    fista.optim(X=sig_val, lmbd=lmbd, Z=zs_val,
                max_iter=2*max_iter, tol=1e-12)
    c_val = min(c_val, fista.train_cost[-1])
    fista.optim(X=sig_test, lmbd=lmbd, Z=zs_t,
                max_iter=2*max_iter, tol=1e-15)
    c_tst = min(c_tst, fista.train_cost[-1])

    # construct feed for test and validation cost
    feed_test = {"Z": zs_test, "X": sig_test, "lmbd": lmbd, "c_tst": c_tst}
    feed_val = {"Z": zs_val, "X": sig_val, "lmbd": lmbd, "c_val": c_val}

    # Reload past experiment if needed
    try:
        curve_cost = np.load(osp.join(SAVE_DIR, "curve_cost.npy"))
        curve_cost_f = np.load(osp.join(SAVE_DIR, "curve_cost_f.npy"))
    except FileNotFoundError:
        curve_cost = np.zeros(len(layer_lvl))
        curve_cost_f = np.zeros(len(layer_lvl))

    wp, wpf = [], []
    lista_nets = [None for _ in layer_lvl]
    lifsta_nets = [None for _ in layer_lvl]
    for i, ((run, run2), k) in enumerate(zip(run_net_lvl, layer_lvl)):
        # ## LISTA
        param_file = osp.join(SAVE_DIR, 'w_lista_{}.npy'.format(k))
        if run:
            listak = LIstaNetwork(D, n_layers=k, shared=False,
                                  warm_param=wp, gpu_usage=gpu_usage)
            # listak.reset()
            # Reload previously trained network
            if reload_net:
                if osp.exists(param_file):
                    wp = np.load(param_file)
                    listak.import_param(wp)

            # Train the network for the given problem
            listak.train(pb, max_iter, steps, feed_test, feed_val, lr_init=lr,
                         reg_cost=8, tol=1e-6, test_cost=True,
                         model_name='{}/lista_{}'.format(NAME_EXP, i))

            curve_cost[i] = listak.cost(**feed_test)
            np.save(osp.join(SAVE_DIR, 'curve_cost.npy'), curve_cost)
            checkpoint(listak, curve_cost, k, SAVE_DIR)

            # Liberate ressources
            listak = None

        # Makes sure warm_param works even without running everything
        if osp.exists(param_file) and warm_param:
            wp = np.load(param_file)

        # ## LFISTA
        param_file = osp.join(SAVE_DIR, 'w_lfista_{}.npy'.format(k))
        if run2:
            lfk = LFistaNetwork(D, n_layers=k, shared=False,
                                warm_param=wpf, gpu_usage=gpu_usage)
            # lfk.reset()
            # Reload previously trained network
            if reload_net:
                if osp.exists(param_file):
                    wp = np.load(param_file)
                    lfk.import_param(wp)

            # Train the network for the given problem
            lfk.train(pb, max_iter, steps, feed_test, feed_val, lr_init=lr,
                      reg_cost=8, tol=1e-6, test_cost=True,
                      model_name='{}/lfista_{}'.format(NAME_EXP, i))
            if warm_param:
                wpf = lfk.export_param()

            # Save the experiment advance
            curve_cost_f[i] = lfk.cost(**feed_test)
            np.save(osp.join(SAVE_DIR, 'curve_cost_f.npy'), curve_cost_f)
            checkpoint(listak, curve_cost, k, SAVE_DIR)

            # Liberate ressources
            lfk = None

        # Makes sure warm_param works even without running everything
        if osp.exists(param_file) and warm_param:
            wpf = np.load(param_file)

    curve_cost_f[0] = curve_cost[0]

    np.save(osp.join(SAVE_DIR, 'curve_cost.npy'), curve_cost)
    np.save(osp.join(SAVE_DIR, 'curve_cost_f.npy'), curve_cost_f)
    # np.save("save_exp/sparsity/final_point.npy", final_point)

    if True:
        save_value = dict(layer_lvl=layer_lvl, curve_cost=curve_cost,
                          curve_cost_f=curve_cost_f, cost_ista=ista.train_cost,
                          cost_fista=fista.train_cost, pb=pb, cost_z0=c_tst,
                          L=np.linalg.norm(D, ord=2)**2)
        import pickle
        with open(osp.join(SAVE_DIR, 'save_layer2_1507.pkl'), 'wb') as f:
            pickle.dump(save_value, f)

    import IPython
    IPython.embed()

    eps = 5e-7
    cost_ista = ista.train_cost
    cost_fista = fista.train_cost

    # Plot the layer curve
    c_star = min(cost_ista[-1], cost_fista[-1], curve_cost_f[-1])-eps
    k = np.arange(1, len(cost_ista)+1)
    plt.loglog(range(1, len(cost_ista)+1), cost_ista-c_star, 'g', label='ISTA')
    plt.loglog(range(1, len(cost_fista)+1), cost_fista-c_star, 'ys-',
               markevery=layer_lvl, label='FISTA')
    plt.loglog(layer_lvl, curve_cost-c_star, 'bo-', label='LISTA')
    plt.loglog(layer_lvl, curve_cost_f-c_star, 'c*-', label='LFISTA')

    plt.hlines(eps, 1, len(cost_ista)+1, 'k', '--')
    plt.legend(loc=3, fontsize='x-large')
    plt.xlim((1, 400))
    plt.ylim((eps*.5, 7))
    plt.xlabel('# iteration/layers k', fontsize='x-large')
    plt.ylabel('Cost function $F(z) - F(z^*)$', fontsize='x-large')

    import IPython
    IPython.embed()
