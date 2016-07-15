# coding: utf-8
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

from Lcod.image_problem_generator import ImageProblemGenerator,\
    create_dictionary_haar
from Lcod.lista_network2 import LIstaNetwork2
from Lcod.lfista_network2 import LFistaNetwork2
from Lcod.ista_tf import IstaTF
from Lcod.fista_tf import FistaTF


# General Setup and constants

# Constants
p = 8                  # Dimension of the patches
N_test = 10000         # Number of test sample
N_val = 1000           # Number of test sample

lmbd = .01             # Regularisation level
corr = 0               # Correlation level for the coefficients

# Setup the training constant and a test set
steps = 100
batch_size = 500
max_iter = 400

SAVE_DIR = osp.join('save_exp', 'layers2')
if not osp.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

layer_lvl = np.unique(np.logspace(0, 2, 10).astype(int))
print("NB layers: {}".format(len(layer_lvl)))
print(layer_lvl)

# Setup the problem
D = create_dictionary_haar(p, wavelet='haar')
L = np.linalg.norm(D, ord=2)**2

pb = ImageProblemGenerator(D, lmbd, batch_size=batch_size,
                           seed=1234)
sig_test, z0_test, zs_test, _ = pb.get_test(N_test)
c_tst = pb.lasso_cost(z0_test, sig_test)

sig_val, z0_val, zs_val, _ = pb.get_batch(N_val)
c_val = pb.lasso_cost(z0_val, sig_val)


def train_ista():
    ista = IstaTF(D)
    zs_t = np.copy(zs_test)
    ista.optimize(X=sig_val, lmbd=lmbd, Z=zs_val,
                  max_iter=2*max_iter, tol=1e-12)
    c_val = ista.train_cost[-1]
    ista.optimize(X=sig_test, lmbd=lmbd, Z=zs_t,
                  max_iter=2*max_iter, tol=1e-12)
    fista = FistaTF(D)
    zs_t = np.copy(zs_test)
    fista.optim(X=sig_val, lmbd=lmbd, Z=zs_val, max_iter=2*max_iter, tol=1e-12)
    c_val = min(c_val, fista.train_cost[-1])
    fista.optim(X=sig_test, lmbd=lmbd, Z=zs_t, max_iter=2*max_iter, tol=1e-12)
    return ista, fista, c_val

ista, fista, c_val_i = train_ista()
c_tst = min(ista.train_cost[-1], fista.train_cost[-1], c_tst)
c_val = min(c_val, c_val_i)
feed_test = {"Z": zs_test, "X": sig_test, "lmbd": lmbd, "c_tst": c_tst}
feed_val = {"Z": zs_val, "X": sig_val, "lmbd": lmbd, "c_val": c_val}

# ###  Influence of the number of layers
# Generate a model for signals

# Reload past experiment if needed
try:
    lista_nets
    lifsta_nets
except NameError:
    lista_nets = []
    lifsta_nets = []
    try:
        curve_cost = np.load(osp.join(SAVE_DIR, "curve_cost.npy"))
        curve_cost_f = np.load(osp.join(SAVE_DIR, "curve_cost_f.npy"))
    except FileNotFoundError:
        curve_cost = np.zeros(len(layer_lvl))
        curve_cost_f = np.zeros(len(layer_lvl))
print(curve_cost_f)

warm_param = False
optim_lvl = [(False, 1e-1, 500, False), (False, 1e-1, 500, False),
             (False, 1e-1, 500, False), (False, 1e-1, 500, False),
             (False, 1e-1, 500, False), (False, 1e-1, 500, False),
             (False, 1e-1, 500, False), (True, 1e-1, 500, True),
             (True, 1e-1, 500, True)]
wp, wpf = [], []
lista_nets = [None for _ in layer_lvl]
lifsta_nets = [None for _ in layer_lvl]
print(layer_lvl)
for i, ((run, lr, m_iter, run2), k) in enumerate(zip(optim_lvl, layer_lvl)):
    if run:
        if lista_nets[i] is None:
            lista_nets[i] = LIstaNetwork2(D, n_layers=k, shared=False,
                                          warm_param=wp)
        listak = lista_nets[i]
        # listak.reset()
        listak.train_cost = listak.train_cost[:-1]
        listak.train(pb, m_iter, steps, feed_test, feed_val, lr_init=lr,
                     reg_cost=8, tol=1e-6, test_cost=True,
                     model_name='layers/lista_{}'.format(i))
        if warm_param:
            wp = listak.export_param()

        curve_cost[i] = listak.cost(**feed_test)
        np.save(osp.join(SAVE_DIR, 'train_cost/lista_{}'.format(k)),
                listak.train_cost)
        np.save(osp.join(SAVE_DIR, 'curve_cost.npy'),
                curve_cost)
        # final_point[i] = listak.output(X=sig_test)
    if run2:
        if lifsta_nets[i] is None:
            lifsta_nets[i] = LFistaNetwork2(D, n_layers=k, shared=False,
                                            warm_param=wpf)
        lfk = lifsta_nets[i]
        lfk.steps = steps
        # lfk.reset()
        lfk.train(pb, m_iter, steps, feed_test, feed_val, lr_init=lr,
                  reg_cost=8, tol=1e-6, test_cost=True,
                  model_name='layers/lfista_{}'.format(i))
        if warm_param:
            wpf = lfk.export_param()

        curve_cost_f[i] = lfk.cost(**feed_test)
        np.save(osp.join(SAVE_DIR, 'train_cost/lfista_{}'.format(k)),
                lfk.train_cost)
        np.save(osp.join(SAVE_DIR, 'curve_cost_f.npy'), curve_cost_f)
        # final_point_f[i] = lfk.output(X=sig_test)

curve_cost_f[0] = curve_cost[0]

np.save(osp.join(SAVE_DIR, 'curve_cost.npy'), curve_cost)
np.save(osp.join(SAVE_DIR, 'curve_cost_f.npy'), curve_cost_f)
# np.save("save_exp/sparsity/final_point.npy", final_point)

if True:
    save_value = dict(layer_lvl=layer_lvl, curve_cost=curve_cost,
                      curve_cost_f=curve_cost_f, cost_ista=ista.train_cost,
                      cost_fista=fista.train_cost, pb=pb, cost_z0=c_tst, L=L)
    import pickle
    with open(osp.join(SAVE_DIR, 'save_layer2_77.pkl'), 'wb') as f:
        pickle.dump(save_value, f)

import IPython
IPython.embed()
