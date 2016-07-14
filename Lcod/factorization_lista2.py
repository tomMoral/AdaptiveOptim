import numpy as np
from sys import stdout as out


class FactorizationLSITA(object):
    """docstring for FactorizationLSITA"""
    def __init__(self, D, lmbd, T=100):
        super().__init__()
        self.D, self.lmbd = D, lmbd
        self.K = D.shape[0]
        self.B = D.dot(D.T)
        self.T = T

    def _init(self, x, zs):
        B = self.B
        U, S, _ = np.linalg.svd(B)
        P, d, _ = np.linalg.svd(U)
        At = [P.dot(np.diag(d**(t/self.T))).dot(P.T)
              for t in range(self.T)]
        St = [self._get_S(A) for A in At]

        metric = [self.cost(A, S, x, zs) for A, S in zip(At, St)]
        i0 = np.argmin(metric, axis=0)
        print(i0)
        return metric[i0[0]], (At[i0[0]], St[i0[0]])

    def factorize(self, x, zs, pb, lr=.1, max_iter=500):

        x_init, zs_init, _ = pb.get_batch()
        x_init, zs_init = x, z
        c, (A, S) = self._init(x_init, zs_init)
        res = [(A, S)]
        cost = [c]
        cost_test = [self.cost(A, S, x, z)]
        for i in range(max_iter):
            out.write('\rFactorize: {:7.2%}'.format(i/max_iter))
            x_train, zs_train, _ = pb.get_batch()
            x_train, zs_train = x, z

            # Gradient step for A
            g = self.grad(A, S, x_train, zs_train)
            A_2 = A - lr*g
            Q, _, P = np.linalg.svd(A_2)
            A = Q.dot(P)

            # retrieve S associated to A
            S = self._get_S(A)
            cost += [self.cost(A, S, x_train, zs_train)]
            cost_test += [self.cost(A, S, x, zs)]
            res += [(A, S)]
        print('\rFactorize:    done')
        i0 = np.argmin(cost, axis=0)
        print(i0)
        return cost, cost_test, res[i0[0]]

    def _get_S(self, A, eps=1e-6):
        X = A.dot(self.B).dot(A.T)
        S = np.sum(abs(X), axis=1)
        S += eps*np.ones(S.shape)
        return S

    def cost(self, A, S, x, zs):
        z1, _ = self._get_z1(A, S, x)
        R = A.dot(S*A.T) - self.B
        l2 = zs.dot(R.dot(zs.T))
        l2 = np.trace(l2)
        l2 = np.sum(zs.dot(R)*zs, axis=1)
        l2 = np.mean(l2)/2

        delta = np.sum(np.abs(A.T.dot(zs.T)), axis=0)
        delta -= np.sum(np.abs(zs.T), axis=0)
        delta -= np.sum(np.abs(A.T.dot(z1.T)), axis=0)
        delta += np.sum(np.abs(z1.T), axis=0)
        delta = delta.mean()
        return [l2 + self.lmbd*delta, self.cost_lasso(x, z1)]

    def cost_lasso(self, x, z1):
        res = x - z1.dot(D)
        l2 = np.mean(np.sum(res*res, axis=1))/2
        l1 = np.mean(abs(z1).sum(axis=1))
        return l2 + self.lmbd*l1

    def grad(self, A, S, x, zs):
        return (- self.lmbd*self._gradient_deltaA(A, S, x, zs) +
                self._gradient_R(A, S, zs))

    def _gradient_deltaA(self, A, S, x, zs):
        z1, dz1 = self._get_z1(A, S, x)
        N = x.shape[0]
        grad = (np.sign(zs.dot(A)).T.dot(zs))
        grad -= (np.sign(z1.dot(A)).T.dot(z1 + dz1.dot(A)))
        grad += (np.sign(z1).T.dot(dz1))
        return grad/N

    def _gradient_R(self, A, S, zs):
        R = S[:, None]*A.T
        return zs.T.dot(zs.dot(R))/zs.shape[0]

    def _get_z1(self, A, S, x, zk=None):
        """Compute a step of the optimization according to the factorization
        A, S provided and starting from the point zk.
        """
        if zk is None:
            zk = np.zeros(shape=(x.shape[0], self.K))

        # y = pinv(A)x and B = D^TD thus B.y = D^Tx
        By = x.dot(self.D.T)

        # Compute prox of the point A.z_k + AS^{-1}B.y
        hk = zk.dot(A) + By.dot(A)/S[None, :]
        shk = np.sign(hk)
        p = shk*np.maximum(0, np.abs(hk) - self.lmbd/S)
        z1 = p.dot(A.T)
        dz1 = (p + By.dot(A.T/S[None, :]))*(abs(p) >= 1e-8)

        return z1, dz1


if __name__ == '__main__':
    N = 500
    N_test = N
    K = 100
    lmbd = 1000
    np.random.seed(3615)
    D = np.random.normal(size=(K, 64))
    L = abs(D.dot(D.T)).sum(axis=-1).max()

    from simple_problem_generator import SimpleProblemGenerator
    pb = SimpleProblemGenerator(D, lmbd, rho=10/K, batch_size=N,
                                seed=4227)
    x, z, _ = pb.get_batch(N_test)

    I = np.eye(K)

    FI = FactorizationLSITA(D, lmbd, T=100)
    c, ct, (Af, Sf) = FI.factorize(x, z, pb, lr=.001, max_iter=50)
    c, ct = np.array(c), np.array(ct)

    Z1, _ = FI._get_z1(I, L*np.ones(K), x)
    Z2, _ = FI._get_z1(I, FI._get_S(I), x)
    Z3, _ = FI._get_z1(Af, Sf, x)
    from utils import soft_thresholding
    z1 = soft_thresholding(x.dot(D.T/L), lmbd/L)

    assert np.allclose(z1, Z1), (
        "Factorize should give the same result as ista for A=I and S = L.I"
    )

    c_ista = FI.cost(I, L*np.ones(K), x, z)
    c_FI = FI.cost(I, FI._get_S(I), x, z)
    print("E(z^*) : ", FI.cost_lasso(x, z))
    print("ISTA: ", FI.cost_lasso(x, z1))
    print("Test: ", FI.cost_lasso(x, Z1))
    print("Fact: ", FI.cost_lasso(x, Z2))
    print("Full fact: ", c[-1][1])
    i = c.argmin(axis=0)
    print("Min fact: ", FI.cost_lasso(x, Z3))
    print("Min fact: ", c[i[0], 1])
    print("Min fact real: ", c[i[1], 1])
    import matplotlib.pyplot as plt
    plt.semilogy(np.arange(1, len(c)+1), c[:, 0])
    plt.semilogy(np.arange(1, len(ct)+1), ct[:, 0])
    plt.hlines([c_ista[0], c_FI[0]], 0, 100)
    # plt.ylim((min(c[:, 1].min(), c_ista)*.97, 1.05*max(c_ista, c[:, 1].max())))
    plt.show()
