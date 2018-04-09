import numpy as np
import pytest

from adaopt.ista_tf import IstaTF
from adaopt.lista_network import LIstaNetwork
from adaopt.lista_network_torch import LIstaTorch
from adaopt.facto_network import FactoNetwork
from adaopt.fista_tf import FistaTF
from adaopt.lfista_network import LFistaNetwork
from adaopt.simple_problem_generator import SimpleProblemGenerator


def pytest_generate_tests(metafunc):
    ids = []
    argnames = ['class_network']
    argvalues = []
    for cn in metafunc.cls.network_classes:
        ids.append(cn.__name__)
        argvalues.append(([cn]))
    metafunc.parametrize(argnames, argvalues, ids=ids, scope="class")


class _TestNetwork:

    @pytest.mark.parametrize('n_layers', [1, 3, 7, 12, 19, 25])
    def test_init(self, n_layers, class_network):
        K = 10
        p = 5
        n = 100
        lmbd = .1
        D = np.random.normal(size=(K, p))

        D /= np.sqrt((D*D).sum(axis=1))[:, None]
        pb = SimpleProblemGenerator(D, lmbd)
        X, _, Z, lmbd = pb.get_batch(n)

        feed_test = {"Z": Z,
                     "X": X,
                     "lmbd": lmbd}

        classic = self.classic_class(D)
        classic.optimize(X, lmbd, Z, max_iter=n_layers, tol=-1)
        classic.terminate()

        network = class_network(D, n_layers=n_layers)
        c = network.cost(**feed_test)
        assert np.isclose(c, classic.train_cost[n_layers])
        network.terminate()


class MixtureIsta:
    classic_class = IstaTF
    network_classes = [
        LIstaNetwork,
        FactoNetwork,
        LIstaTorch
    ]


class MixtureFista:
    network_classes = [LFistaNetwork]
    classic_class = FistaTF


class TestFista(_TestNetwork, MixtureFista):
    pass


class TestIsta(_TestNetwork, MixtureIsta):
    pass