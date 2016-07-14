import numpy as np


def n2(x):
    return np.sum(x*x)


def soft_thresholding(x, theta):
    '''Return the soft-thresholding of x with theta
    '''
    return np.sign(x)*np.maximum(0, np.abs(x) - theta)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    t = np.linspace(-1, 1, 1000)
    plt.plot(t, soft_thresholding(t, .2))
    plt.plot(t, [n2(tt) for tt in t])
    plt.show()
