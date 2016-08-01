import numpy as np


def softhresh(inp, t):
    return np.sign(inp)*np.maximum(0, abs(inp) - t)


def prod0bp(A, B, din):
    return 2 * din.dot(A.dot(B))


def resbp(A, S, din):
    dA = 2 * din.dot(A.T.dot(S))
    dS = A.dot(din.dot(A.T))
    dS = np.diag(np.diag(dS))
    return dA, dS


def tracebp(R, Z):
    return Z.dot(Z.T) / Z.shape[1]
    # return np.sum([np.outer(z, z) for z in Z.T], axis=0)


def commbp(A, Z, Z1, lmbd):
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


def cost(sig, D, z, lmbd):
    res = sig - D.dot(z)
    l2 = np.sum(res*res, axis=0)
    l1 = abs(z).sum(axis=0)
    return (l2/2 + lmbd*l1).mean()


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('Test for matrix factorization')
    parser.add_argument('--update_AS', action='store_true',
                        help='Update independantly A and S. If false, all the '
                             'gradient is computed for X st R = X^TX.')

    args = parser.parse_args()

    N = 64
    M = 128
    L = 2000
    rho = .1  # 10/M
    lmbd = 0.1
    beta = 2.5
    resid = 0

    D = np.random.normal(size=(N, M))
    D = D/np.sqrt(np.sum(D*D, axis=0))[None, :]

    B = D.T.dot(D)
    nB = np.linalg.norm(B, ord=2)

    Z = np.random.normal(size=(M, L))*(np.random.random(size=(M, L)) < rho)
    X = D.dot(Z)
    lmbd = lmbd*np.mean(np.sum(X*X, axis=0))

    ZS = np.zeros(Z.shape)
    from sys import stdout as out
    for i in range(1000):
        ZS1 = ZS
        ZS = softhresh(ZS1-1/nB*(B.dot(ZS1)) + 1/nB*D.T.dot(X), lmbd/nB)
        dz = np.sqrt(np.sum((ZS-ZS1)**2))
        out.write('\r{:6.2%} - {:.4e}'.format(i/1000, dz))
        out.flush()
        if dz < 1e-5:
            print("\rNeeded {} iteration of ISTA".format(i))
            break
    C0 = cost(X, D, ZS, lmbd)

    _, S, A = np.linalg.svd(B)
    S = np.diag(S)

    K = S.max()
    XX = np.linalg.pinv(D).dot(X)
    Z1 = np.sign(XX)*np.maximum(0, abs(XX) - lmbd/K)
    disti = np.sqrt(np.mean((ZS-Z1)**2, axis=0))[None, :]
    Znoise = ZS + disti*np.random.normal(size=Z.shape)

    def test_facto(A, S):
        pS = np.linalg.pinv(S)
        Z2 = A.T.dot(softhresh((A-pS.dot(A.dot(B))).dot(Z1) +
                               pS.dot(A.dot(D.T.dot(X))),
                               lmbd/np.diag(S)[:, None]))
        Z2b = softhresh(Z2-1/nB*(B.dot(Z2)) + 1/nB*D.T.dot(X), lmbd/nB)

        Z3 = softhresh(Z1-1/nB*B.dot(Z1) + 1/nB*D.T.dot(X), lmbd/nB)
        Z3b = softhresh(Z3-1/nB*B.dot(Z3)+1/nB*D.T.dot(X), lmbd/nB)

        errore2 = cost(X, D, Z2, lmbd)-C0

        errore3 = cost(X, D, Z3, lmbd)-C0

        print("Facto:", errore2, "ISTA:", errore3)
        print("Improvement: {:7.2%}".format((errore3-errore2)/errore3))
        return (errore3-errore2)/errore3

    tatref = lmbd*(np.sum(np.maximum(0, np.sum(
        abs(A.dot(ZS))-abs(ZS)-abs(A.dot(Znoise))+abs(Znoise), axis=0))))
    Rref = nB*np.eye(M) - B
    tatref2 = np.sum([z.T.dot(Rref.dot(z)) for z in (ZS-Z1).T])/2

    print("Ref:", tatref)
    print("Ref2:", tatref2)

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

    niters = 20000
    lr = 1e-2
    mom = 0
    momS = 0
    momRR = 0
    update_AS = args.update_AS

    rhomom = .8
    rhomomS = .8
    rhomomRR = .8
    trt, tat, tet, err = [], [], [], []
    dr = tracebp(None, ZS-Z1)
    dR = np.zeros(dr.shape)

    test = [test_facto(A, S)]

    print("Cost Z0:", C0)
    print("Cost 0:", cost(X, D, np.zeros(Z.shape), lmbd))
    print("Cost Z1:", cost(X, D, Z1, lmbd))
    n_ds = 5
    for n in range(1, niters):
        if n % (niters//n_ds) == (niters//n_ds-1):
            print("downscaling the learning rate")
            lr *= 0.7

        if n % (niters//100) == 0:
            test += [test_facto(A, S)]
            if test[-1] < test[-2] and n > niters//100:
                print("Down scale momentum")
                # mom = momS = momRR = 0
                lr *= .9
                rhomomRR *= .9
                rhomom = rhomomS = rhomomRR

        dA = commbp(A, ZS, Znoise, lmbd)
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

        trt += [np.mean([z.T.dot(Rt.dot(z)) for z in (ZS-Z1).T])/2]
        tat += [lmbd*(np.mean(np.sum(
            abs(A.dot(ZS)) - abs(ZS) - abs(A.dot(Znoise)) + abs(Znoise),
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
            Rt = A.T.dot(S.dot(A)) - B

        # Update RR
        momRR = rhomomRR * momRR - lr * dRR
        RR += momRR

        if not update_AS:
            # Compute new matrix
            Rt = RR.T.dot(RR)
            _, S, A = np.linalg.svd(Rt+B)
            S = np.diag(S)

        if n % np.ceil(niters/500) == 1:
            print('Iterations {:7.2%}:  {} - {:.4e} - {:.4e} - {:.4e}'
                  .format((n-1)/niters, err[-1], tat[-1], trt[-1], tet[-1]))

    Afin = A
    Sfin = S

    test_facto(Afin, Sfin)

    import matplotlib.pyplot as plt
    plt.plot(err)
    plt.show()

    import IPython
    IPython.embed()
