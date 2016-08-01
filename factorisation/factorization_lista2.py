import numpy as np
from sys import stdout as out

from .utils import soft_thresholding


class FactorizationLISTA(object):
    """docstring for FactorizationLSITA"""
    def __init__(self, D, lmbd, T=100):
        super().__init__()
        self.D, self.lmbd = D.T, lmbd
        self.K = D.shape[0]
        self.B = D.dot(D.T)
        self.L = np.linalg.norm(self.B, ord=2)
        self.T = T

    def _init(self, x, zs):
        B = self.B
        M = B.shape[0]
        nB = np.linalg.norm(B, ord=2)
        _, S, A = np.linalg.svd(B)
        S = np.diag(S)

        K = S.max()
        XX = np.linalg.pinv(self.D).dot(x)
        z1 = np.sign(XX)*np.maximum(0, abs(XX) - lmbd/K)
        disti = np.sqrt(np.mean((zs-z1)**2, axis=0))[None, :]
        Znoise = zs + disti*np.random.normal(size=zs.shape)

        tatref = self.lmbd*(np.sum(np.maximum(0, np.sum(
            abs(A.dot(zs))-abs(zs)-abs(A.dot(Znoise))+abs(Znoise), axis=0))))
        Rref = nB*np.eye(M) - B
        tatref2 = np.sum([z.T.dot(Rref.dot(z)) for z in (zs-z1).T])/2

        if tatref < tatref2:
            noise = 0.00001
            RR = noise*np.random.normal(size=(M, M))
            Rt = RR.T.dot(RR)
            _, S, A = np.linalg.svd(B+Rt)
            S = np.diag(S)
            print('init from svd')
        else:
            A = np.eye(M)
            S = nB*np.eye(M)
            Rt = A.T.dot(S.dot(A)) - B
            ll, V = np.linalg.eigh(Rt)
            RR = np.diag(np.sqrt(np.maximum(0, ll))).dot(V.T)
            # Rt = RR.T.dot(RR)
            print('init from identity')

        return tatref, A, S, RR, Rt

    def factorize(self, x, zs, zk, lr=2e-2, niters=2000, beta=2.5):
        x = x.T
        zs = zs.T
        zk = zk.T
        disti = np.sqrt(np.mean((zs-zk)**2, axis=0))[None, :]
        znoise = zs + disti*np.random.normal(size=zs.shape)
        noise = 0.000001
        RR = noise*np.random.normal(size=(self.K, self.K))
        Rt = RR.T.dot(RR)
        _, S, A = np.linalg.svd(self.B+Rt)
        S = np.diag(S)

        C0 = self.cost(x, zs)

        mom = 0
        momS = 0
        momRR = 0
        update_AS = True

        rhomom = .6
        rhomomS = .6
        rhomomRR = .6
        trt, tat, tet, err = [], [], [], []
        dr = self.tracebp(None, zs-zk)
        dR = np.zeros(dr.shape)

        test = [self.test_facto(A, S, x, zk, C0)]

        print("Cost Z0:", C0)
        print("Cost 0:", self.cost(x, np.zeros(zs.shape)))
        print("Cost Z1:", self.cost(x, zk))
        n_ds = 5
        for n in range(1, niters):
            if n % (niters//n_ds) == (niters//n_ds-1):
                print("downscaling the learning rate")
                lr *= 0.7

            if n % (niters//100) == 0:
                test += [self.test_facto(A, S, x, zk, C0)]
                if test[-1] < test[-2] and n > niters//100:
                    print("Down scale momentum")
                    # mom = momS = momRR = 0
                    lr *= .9
                    rhomomRR *= .9
                    rhomom = rhomomS = rhomomRR

            dA = self.commbp(A, zs, znoise, self.lmbd)
            dR[:] = dr  # tracebp(Rt, ZS-Z1)
            if update_AS:
                resid = (Rt - RR.T.dot(RR))
                dR += beta*resid
                dRR = -2*beta*RR.dot(resid)

                dS = np.diag(np.diag(A.dot(dR.dot(A.T))))
                dA += 2*S.dot(A.dot(dR))
                dA -= A.dot(dA.T.dot(A))  # Keep update unitary

            else:
                # dR2 = A.T.dot(S.dot(dA)) + dA.T.dot(S.dot(A))
                # dR += dR2
                # dRR = np.linalg.pinv(RR.T, rcond=1e-5).dot(dR)
                dRR = RR.dot(dr)
                dA -= A.dot(dA.T.dot(A))  # Keep update unitary

            trt += [np.mean([z.T.dot(Rt.dot(z)) for z in (zs-zk).T])/2]
            tat += [self.lmbd*(np.mean(np.sum(
                abs(A.dot(zs)) - abs(zs) - abs(A.dot(znoise)) + abs(znoise),
                axis=0)))]
            tet += [beta*np.sum(resid*resid)/2*update_AS]
            err += [trt[-1] + tat[-1] + tet[-1]]

            if update_AS:
                # Update A and S
                mom = rhomom * mom - lr * dA
                A += mom
                U2, S2, V2 = np.linalg.svd(A)
                A = U2.dot(V2)
                momS = rhomomS * momS - lr * dS
                S += momS

                # Compute new matrix
                Rt = A.T.dot(S.dot(A)) - self.B

            # Update RR
            momRR = rhomomRR * momRR - lr * dRR
            RR += momRR

            if not update_AS:
                # Compute new matrix
                Rt = RR.T.dot(RR)
                _, S, A = np.linalg.svd(Rt+self.B)
                S = np.diag(S)

            if n % np.ceil(niters/500) == 1:
                print('Iterations {:7.2%}:  {} - {:.4e} - {:.4e} - {:.4e}'
                      .format((n-1)/niters, err[-1], tat[-1],
                              trt[-1], tet[-1]))

        return A.T, np.diag(S)

    def _get_S(self, A, eps=1e-6):
        X = A.dot(self.B).dot(A.T)
        S = np.sum(abs(X), axis=1)
        S += eps*np.ones(S.shape)
        return S

    def _cost(self, A, S, x, zs):
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

    def cost(self, sig, z):
        res = sig - self.D.dot(z)
        l2 = np.sum(res*res, axis=0)
        l1 = abs(z).sum(axis=0)
        return (l2/2 + self.lmbd*l1).mean()

    def test_facto(self, A, S, x, zk, C0):
        B, D = self.B, self.D
        pS = np.linalg.pinv(S)
        nB = self.L

        Z_facto = A.T.dot(soft_thresholding(
            (A-pS.dot(A.dot(B))).dot(zk) + pS.dot(A.dot(D.T.dot(x))),
            self.lmbd/np.diag(S)[:, None]))

        Z_ista = soft_thresholding(zk-1/nB*B.dot(zk) + 1/nB*D.T.dot(x),
                                   self.lmbd/nB)

        errore2 = self.cost(x, Z_facto)-C0

        errore3 = self.cost(x, Z_ista)-C0

        print("Facto:", errore2, "ISTA:", errore3)
        print("Improvement: {:7.2%}".format((errore3-errore2)/errore3))
        return (errore3-errore2)/errore3


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
