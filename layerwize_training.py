import os
import numpy as np
import matplotlib.pyplot as plt

if os.path.exists(os.path.join('/proc', 'acpi', 'bbswitch')):
    # assert that the graphic card is on if bbswitch is detected
    import os
    assert 'BUMBLEBEE_SOCKET' in os.environ.keys()

try:
    import sys
    sys.path.remove("/usr/lib/python3/dist-packages")
except ValueError:
    pass

from Lcod.lista_network import LIstaNetwork
from Lcod.ista_tf import IstaTF


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('Experiments for Adaopt paper')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='If set, the experiments will be saved in the '
                             'specified directory')
    parser.add_argument('--gpu', type=float, default=.95,
                        help='Ratio of usage of the gpu for this launch')
    parser.add_argument('--lmbd', type=float, default=.01,
                        help='Lambda used for the experiments, control the '
                             'regularization level')
    parser.add_argument('--rho', type=float, default=.05,
                        help='Rho used for the experiments, control the '
                             'sparsity level')
    parser.add_argument('--debug', '-d', type=int, default=20,
                        help='Logging level, default is INFO, '
                             'set 10 for DEBUG')
    parser.add_argument('-K', type=int, default=100,
                        help='Number of dictionary elements used.')
    parser.add_argument('--n_layers', type=int, default=20,
                        help='Use the output of this layer to train the '
                        'network.')
    args = parser.parse_args()

    # General Setup and constants
    # Experiment metadata
    save_exp = args.save_dir is not None
    NAME_EXP = args.save_dir if args.save_dir else 'default'
    gpu_usage = args.gpu  # GPU memory allocated to this process
    log_lvl = args.debug  # verbosity -- INFO

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
    lr_init = 5e-2          # Initial learning rate for the gradient descent
    steps = 100             # Number of steps for GD between validation
    batch_size = 300        # Size of the batch for the training

    # Setup the training constant and a test set

    from Lcod.simple_problem_generator import SimpleProblemGenerator
    from Lcod.simple_problem_generator import create_dictionary
    p = 64                 # Dimension of the data
    D = create_dictionary(K, p, seed=290890)
    pb = SimpleProblemGenerator(D, lmbd, rho=rho, batch_size=batch_size,
                                corr=corr, seed=422742)

    sig_test, z0_test, zs_test, _ = pb.get_test(N_test)
    sig_val, z0_val, zs_val, _ = pb.get_batch(N_val)
    C0 = pb.lasso_cost(zs_test, sig_test)

    # Compute optimal values for validation/test sets using ISTA/FISTA
    ista = IstaTF(D, gpu_usage=gpu_usage)
    ista.optimize(X=sig_val, lmbd=lmbd, Z=zs_val,
                  max_iter=10000, tol=1e-8 * C0)
    c_val = ista.train_cost[-1]
    ista.optimize(X=sig_test, lmbd=lmbd, Z=zs_test,
                  max_iter=10000, tol=1e-8 * C0)

    feed_test = {"Z": zs_test, "X": sig_test, "lmbd": lmbd}
    feed_val = {"Z": zs_val, "X": sig_val, "lmbd": lmbd,
                "c_val": c_val - 1e-10}

    # Free the ressources
    ista.terminate()
    cc = np.array(ista.train_cost)
    c_star = cc[-1] - 1e-6

    # Run the experiments
    layerwise_network = LIstaNetwork(
        D, n_layers=100, shared=False, log_lvl=log_lvl, gpu_usage=gpu_usage,
        reg_scale=reg_scale)
    full_network = LIstaNetwork(
        D, n_layers=100, shared=False, log_lvl=log_lvl, gpu_usage=gpu_usage,
        reg_scale=reg_scale)

    cc_lista = []
    n_layers = args.n_layers
    for k in range(1, n_layers + 1):
        print(f"\n\nTraining layer {k}")
        layerwise_network.train_layer(k, k_cost=n_layers, batch_provider=pb,
                                      feed_val=feed_val, max_iter=800,
                                      steps=steps, reg_cost=8, tol=1e-5,
                                      lr_init=lr_init)
        cc_lista.append(np.array(layerwise_network.curve_cost(**feed_test)))
        layerwise_network._scale_lr = 1

    cc_full = []
    for k in range(1, n_layers + 1):
        print(f"\n\nTraining layer {k}")
        full_network.train_layer(k, batch_provider=pb, feed_val=feed_val,
                                 max_iter=800, steps=steps, reg_cost=8,
                                 tol=1e-5, lr_init=lr_init,
                                 prev=True)
        cc_full.append(np.array(full_network.curve_cost(**feed_test)))
        layerwise_network._scale_lr = 1

    curves = {
        "ista": cc,
        "lista": cc_full,
        "greedy": cc_lista
    }
    np.save("compare_{}".format(n_layers), curves)

    import IPython
    IPython.embed()

    plt.figure("Compare")
    plt.loglog(cc - c_star, label="ISTA")
    plt.loglog(cc_full[-1] - c_star, label="LISTA$_{{{}}}$".format(n_layers))
    plt.loglog(cc_lista[-1] - c_star, label="LISTA$_{{{}}}$ - greedy"
               .format(n_layers))

    plt.legend()
    plt.figure("layerwise")
    plt.loglog(cc - c_star)
    for k, c in enumerate(cc_lista):
        plt.loglog(c - c_star, label=k)

    plt.figure("full")
    plt.loglog(cc - c_star)
    for k, c in enumerate(cc_lista):
        plt.loglog(c - c_star, label=k)
    plt.show()
