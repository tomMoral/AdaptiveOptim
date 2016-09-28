import numpy as np


def create_gaussian_dictionary(K, p, seed=None):
    np.random.seed(seed)
    D = np.random.normal(size=(K, p)).astype(np.float32)
    D /= (D*D).sum(axis=1)[:, None]
    D = D.astype(np.float32)
    return D


def create_adversarial_dictionary(K, p, seed=None):

    np.random.seed(seed)

    sigma = 1e-1

    II = np.random.permutation(K)
    vec = np.zeros(K)
    D = []

    for k in range(p):
        vec = 0*vec
        vec[II[k]] = 1 + sigma*np.random.rand()
        D += [np.real(np.fft.ifft(vec))]

    return np.array(D)


def create_haar_dictionary(p=8):
    import pywt
    c = pywt.wavedec2(np.zeros((p, p)), 'haar')
    D = []
    for k in range(1, len(c)):
        for i in range(3):
            ck = c[k][i]
            l = ck.shape[0]
            for j in range(l):
                for m in range(l):
                    ck[j, m] = 1
                    D += [pywt.waverec2(c, 'haar')]
                    ck[j, m] = 0
    ck = c[0]
    l = ck.shape[0]
    for j in range(l):
        for m in range(l):
            ck[j, m] = 1
            D += [pywt.waverec2(c, 'haar')]
            ck[j, m] = 0
    D = np.array(D).reshape(-1, p*p)

    Dn = []
    for i in range(15):
        Dn += _translate(D[i].reshape((p, p)))
    return np.array(Dn).reshape((-1, p*p))


def _translate(d):
    D1 = []
    for u in range(-4, 5, 2):
        for v in range(-4, 5, 2):
            d1 = np.zeros((8, 8))
            ws1, we1 = max(0, u), u if u < 0 else None
            ws, we = (-u, None) if u <= 0 else (None, -u)
            hs1, he1 = (0, v) if v < 0 else (v, None)
            hs, he = (-v, None) if v <= 0 else (None, -v)
            d1[ws1:we1, hs1:he1] = d[ws:we, hs:he]
            D1 += [d1]
    return D1
