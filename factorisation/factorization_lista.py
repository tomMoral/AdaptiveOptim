import numpy as np
from sys import stdout as out

from utils import soft_thresholding


class FactorizationLISTA(object):
    """docstring for FactorizationLSITA"""
    def __init__(self, D, lmbd, T=100):
        super().__init__()
        self.D, self.lmbd = D, lmbd
        self.K = D.shape[0]
        self.B = D.dot(D.T)
        self.T = T

    def _init(self, x, zs):
        B = self.B
        _, S, A = np.linalg.svd(B)
        # d, P = np.linalg.eig(A)
        Q, d, P = np.linalg.svd(A)
        At = [Q.dot((d**(t/self.T))*P) for t in range(self.T+1)]
        St = [self._get_S(A) for A in At]

        metric = [self.cost(A, S, x, zs) for A, S in zip(At, St)]
        i0 = np.argmin(metric, axis=0)
        print(i0)

        # import matplotlib.pyplot as plt
        # plt.plot(np.array(metric)[:, 0])
        # plt.show()
        return metric[i0[0]], (At[i0[0]], St[i0[0]])

    def factorize(self, pb, lr=.05, N_batch=1000, max_iter=500, test_set=None):

        x_init, zs_init, _ = pb.get_batch(N_batch)

        c, (A, S) = self._init(x_init, zs_init)
        res = [(A, S)]
        cost = [c]
        cost_test = [self.cost(A, S, test_set[0], test_set[1])]
        for i in range(max_iter):
            x, zs, _ = pb.get_batch(N_batch)

            # Gradient step for A
            g, z1 = self.grad(A, S, x, zs)
            c0 = self.delta_A(A, zs, z1)
            c0 += self.delta_R(A, S, zs)
            A_2 = A - lr*g
            c01 = self.delta_A(A_2, zs, z1)
            c01 += self.delta_R(A_2, S, zs)
            if c0 < c01:
                print("\nBad gradient?!\n\n\n")
            Q, d, P = np.linalg.svd(A_2)
            A = Q.dot(P)
            A_2 -= A
            l2 = np.sqrt(np.sum(A_2*A_2))/2
            c1 = self.delta_A(A, zs, z1)
            c1 += self.delta_R(A, S, zs)

            if c0 < c1:
                print("\nBad transform!", c0-c1, l2)

            # compute associated S
            S1 = S
            S = self._get_S(A)
            dS = S1-S
            c2 = self.delta_A(A, zs, z1)
            c2 += self.delta_R(A, S, zs)
            if c2 > c0:
                print("\nBad S", c0-c2, np.sqrt(np.sum(dS*dS)))

            # Store training and test cost
            cost += [self.cost(A, S, x, zs)]
            cost_test += [self.cost(A, S, test_set[0], test_set[1])]
            res += [(A, S)]

            if cost[-1][0] > c0:
                print("Bad step!")

            # Display progress
            out.write('\rFactorize: {:7.2%} - {:5e}'
                      ''.format(i/max_iter, np.sum(g*g)))
            out.flush()
        print('\rFactorize:    done')
        i0 = np.argmin(cost, axis=0)
        print(i0)
        return cost, cost_test, res[i0[0]]

    def _get_S(self, A, eps=1e-6):
        X = A.dot(self.B).dot(A.T)
        K = X.shape[0]

        # for i, sum_j|X_ij|
        rho_i = np.sum(np.abs(X), axis=1)
        S = rho_i + eps

        assert S.shape == (K, )

        return S

    def cost(self, A, S, x, zs):
        z1 = self._get_z1(A, S, x)
        l2 = self.delta_R(A, S, zs)
        delta = self.delta_A(A, zs, z1)

        # also return the lasso cost as we computed z1
        cz1 = self.cost_lasso(x, z1)
        return [l2 + delta, cz1]

    def delta_A(self, A, zs, z1):
        # |Az*| - |z*| - (|Az| - |z|)
        delta = np.sum(np.abs(A.dot(zs.T)), axis=0)
        delta -= np.sum(np.abs(zs.T), axis=0)
        delta -= np.sum(np.abs(A.dot(z1.T)), axis=0)
        delta += np.sum(np.abs(z1.T), axis=0)

        # Return the mean over the samples
        return self.lmbd*delta.mean()

    def delta_R(self, A, S, zs):
        R = A.T.dot(np.diag(S).dot(A)) - self.B
        l2 = np.mean([z.dot(R.dot(z)) for z in zs])
        assert l2 > 0
        return l2/2

    def cost_lasso(self, x, z1):
        res = x - z1.dot(self.D)
        l2 = np.mean(np.sum(res*res, axis=1))/2
        l1 = np.mean(abs(z1).sum(axis=1))
        return l2 + self.lmbd*l1

    def grad(self, A, S, x, zs):
        grad, z1 = self._gradient_deltaA(A, S, x, zs)
        grad += self._gradient_R(A, S, zs)

        # Gradient check
        eps = 1e-2
        c0 = self.delta_A(A, zs, z1)
        c0 += self.delta_R(A, S, zs)
        A1 = A - eps*grad/N
        c1 = self.delta_A(A1, zs, z1)
        c1 += self.delta_R(A1, S, zs)
        if c0 < c1:
            print("\nFull Gradient pas bon!!!")

        return grad, z1

    def _gradient_deltaA(self, A, S, x, zs):
        z1 = self._get_z1(A, S, x)
        N = x.shape[0]

        grad = np.sign(A.dot(zs.T)).dot(zs)
        grad -= np.sign(A.dot(z1.T)).dot(z1)

        # Gradient checker
        eps = 1e-4
        c0 = self.delta_A(A, zs, z1)
        # z2 = self._get_z1(A, S, x)
        z2 = z1
        A1 = A - eps*grad/N
        c1 = self.delta_A(A1, zs, z2)
        if c0 < c1:
            print("\nGradient A pas bon!!!")

        return self.lmbd*grad/N, z1

    def _gradient_R(self, A, S, zs):
        SA = S[:, None]*A
        AB = A.dot(self.B)
        N = zs.shape[0]
        kro_z = zs.T.dot(zs)/N
        grad = SA.dot(kro_z)/N

        # Gradient % S
        X = AB.dot(A.T)
        sX = np.sign(X)*(abs(X) > 1e-10)
        Az = A.dot(zs.T).mean(axis=1)

        # ds1 = np.diag(A.T.dot(kro_z).dot(A))
        # ds2 = sX.dot(AB)
        # gS = ds1[:, None]*ds2
        # gS += sX.T.dot(ds1[:, None]*AB)
        gS = sX.T.dot((Az**2)[:, None]*AB)
        gS += (Az[:, None]**2)*(sX.dot(AB))
        grad += gS/2

        # Proposition Joan:

        # # Gradient checker
        eps = 1e-8
        K = A.shape[0]
        c0 = self.delta_R(A, S, zs)
        G = np.zeros(A.shape)
        for i in range(K):
            for j in range(K):
                A[i, j] += eps
                S1 = self._get_S(A)
                c1 = self.delta_R(A, S1, zs)
                G[i, j] = (c1-c0)/eps
                A[i, j] -= eps
        # import IPython
        # IPython.embed()

        c0 = self.delta_R(A, S, zs)
        eps = 1e-16
        A1 = A - eps*G
        S1 = self._get_S(A1)
        c1 = self.delta_R(A1, S1, zs)
        assert c1 < c0, ("\nGradient R pas bon!!! {} __ {}".format(c0, c1))
        print(c0-c1)
        return grad

    def _get_z1(self, A, S, x, zk=None):
        """Compute a step of the optimization according to the factorization
        A, S provided and starting from the point zk.
        """
        if zk is None:
            zk = np.zeros(shape=(x.shape[0], self.K))

        # y = pinv(A)x and B = D^TD thus B.y = D^Tx
        By = x.dot(self.D.T)

        S1A = A/S[:, None]

        # Compute prox of the point A.z_k + AS^{-1}B.y
        hk = zk.dot(A.T) + By.dot(S1A.T)
        p = soft_thresholding(hk, self.lmbd/S)
        z1 = p.dot(A)

        # dz1 = (p + (By/S).dot(A))*(abs(p) >= 1e-8)

        return z1  # , dz1



    def prod0bp(self, A, B, din):
        return 2 * din.dot(A.dot(B))


    def resbp(self, A, S, din):
        dA = 2 * din.dot(A.T.dot(S))
        dS = A.dot(din.dot(A.T))
        dS = np.diag(np.diag(dS))
        return dA, dS


    def tracebp(self, R, Z):
        return Z.dot(Z.T) / Z.shape[1]
        # return np.sum([np.outer(z, z) for z in Z.T], axis=0)


    def commbp(self, A, Z, Z1, lmbd):
        t0 = np.sum(abs(A.dot(Z)) - abs(Z) - abs(A.dot(Z1)) + abs(Z1), axis=0)
        I0 = (t0 > 0)  # [None, :]

        if np.any(I0):
            t1 = np.sign(A.dot(Z[:, I0])).dot(Z[:, I0].T)
            t2 = np.sign(A.dot(Z1[:, I0])).dot(Z1[:, I0].T)
            # t1 = np.sum([np.outer(z, np.sign(z.dot(A.T))).T
            #              for z, i0 in zip(Z.T, I0) if i0], axis=0)
            # t2 = np.sum([np.outer(z, np.sign(z.dot(A.T))).T
            #              for z, i0 in zip(Z1.T, I0) if i0], axis=0)
        else:
            return np.zeros(A.shape)
        # t1 = np.sum([np.outer(z, i0*np.sign(z.dot(A.T)))
        #              for z, i0, k in zip(Z.T, I0, range(len(I0)))
        #              if k < 1 or i0], axis=0)
        # t2 = np.sum([np.outer(z, i0*np.sign(z.dot(A.T)))
        #              for z, i0, k in zip(Z1.T, I0, range(len(I0)))
        #              if k < 1 or i0], axis=0)
        return lmbd*(t1 - t2)/Z.shape[1]


    def cost(self, sig, D, z, lmbd):
        res = sig - D.dot(z)
        l2 = np.sum(res*res, axis=0)
        l1 = abs(z).sum(axis=0)
        return (l2/2 + lmbd*l1).mean()


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('Factoriization test')
    parser.add_argument('--facto', action='store_true',
                        help='Compute and dispaly the factorization')
    parser.add_argument('--ip', action='store_true',
                        help='Open an Ipython console at the end')
    args = parser.parse_args()

    N = 500
    N_test = 500
    K = 100
    lmbd = .01
    # np.random.seed(3615)
    D = np.random.normal(size=(K, 64))
    L = np.linalg.norm(D, ord=2)**2

    from simple_problem_generator import SimpleProblemGenerator

    if args.facto:
        # from ista_tf import IstaTF
        pb = SimpleProblemGenerator(D, lmbd, rho=10/K, batch_size=N,
                                    seed=None)  # 4227)
        sig_test, z0_test, _ = pb.get_batch(N_test)

        I = np.eye(K)

        FI = FactorizationLISTA(D, lmbd, T=10)
        c, ct, (A, S) = FI.factorize(pb, lr=1e-10, N_batch=N,
                                     max_iter=100,
                                     test_set=(sig_test, z0_test))
        c, ct = np.array(c), np.array(ct)

        # ista = IstaTF(D, lmbd, tol=1e-2)
        # zi = ista.optim(X=sig_test, max_iter=1000)

        Z1 = FI._get_z1(I, L*np.ones(K), sig_test)
        Z2 = FI._get_z1(I, FI._get_S(I), sig_test)
        Z3 = FI._get_z1(A, S, sig_test)
        z1 = soft_thresholding(sig_test.dot(D.T/L), lmbd/L)

        assert np.allclose(z1, Z1), (
            "Factorize should give the same result as ista for A=I and S = L.I"
        )

        c_ista = FI.cost(I, L*np.ones(K), sig_test, z0_test)
        c_ista1 = FI.cost_lasso(sig_test, z1)
        c_star = FI.cost_lasso(sig_test, z0_test)
        print("Z^*: ", c_star)
        print("ISTA1: ", c_ista1)
        # print("ISTAc: ", FI.cost_lasso(sig_test, zi))
        print("Test: ", c_ista[1])
        print("Fact A=I : ", FI.cost_lasso(sig_test, Z2))
        print("Fact: ", FI.cost_lasso(sig_test, Z3))
        i = ct.argmin(axis=0)
        print("Min fact: ", ct[i[0], 1])
        print("Min fact real: ", ct[i[1], 1])
        import matplotlib.pyplot as plt

        plt.figure("Cost facto")
        plt.plot(np.arange(1, len(c)+1), c[:, 0], label='Cost train')
        plt.plot(np.arange(1, len(ct)+1), ct[:, 0], label='Cost test')
        # plt.hlines(c_ista[0], 1, len(ct)+1, 'r', '--')
        # plt.ylim((.97*min(c[:, 0].min(), ct[:, 0].min()),
        #           1.05*max(c[:, 0].max(), ct[:, 0].max(), c_ista[0])))
        plt.ylim((.9*min(ct[:, 0]), 1.1*max(ct[:, 0])))

        plt.figure("Cost lasso")
        plt.semilogy(np.arange(1, len(c)+1), c[:, 1], label='Cost train')
        plt.semilogy(np.arange(1, len(ct)+1), ct[:, 1], label='Cost test')
        # plt.hlines(c_ista[1], 1, len(ct)+1, 'r', '--')
        # plt.hlines(c_star, 1, len(ct)+1, 'k', '--')
        # plt.ylim((.97*min(c[:, 1].min(), ct[:, 1].min(), c_star),
        #           1.05*max(c[:, 1].max(), ct[:, 1].max(), c_ista[1])))
        # plt.legend()
        plt.ylim((.9*min(ct[:, 1]), 1.1*max(ct[:, 1])))
        plt.show()

    if args.ip:
        import IPython
        IPython.embed()
