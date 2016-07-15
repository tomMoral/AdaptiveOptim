import numpy as np
from glob import glob
from scipy.misc import imresize
from tensorflow.examples.tutorials.mnist import input_data


class MnistProblemGenerator(object):
    """A simple problem to test the capability of LISTA

    Parameters
    ----------
    D: array-like, [K, p]
        dictionary for the generation of problems. The shape should be
        [K, p] where K is the number of atoms and p the dimension of the
        output space
    lmbd: float
        sparsity factor for the computation of the cost
    """
    def __init__(self, D, lmbd, batch_size=100,
                 seed=None):
        super().__init__()
        self.D = np.array(D)
        self.K, self.p = D.shape
        self.lmbd = lmbd
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        self.patch_size = int(np.sqrt(self.p))

        # Load training images
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    def get_batch(self, N=None):
        '''Generate a set of N problems, with a signal, a starting point and
        the waited answer.
        '''
        if N is None:
            N = self.batch_size

        # Take mnist 17x17
        im = self.mnist.train.next_batch(N)[0].reshape(N, 28, 28)
        im = [imresize(a, (17, 17), interp='bilinear', mode='L')-.5
              for a in im]
        z = np.zeros((N, self.K))
        X = np.array(im).reshape(N, -1)
        return X, z, z, self.lmbd

    def get_test(self, N=None):
        '''Generate a set of N problems, with a signal, a starting point and
        the waited answer.
        '''
        if N is None:
            N = self.batch_size

        # Take mnist 17x17
        im = self.mnist.test.next_batch(N)[0].reshape(N, 28, 28)
        im = [imresize(a, (17, 17), interp='bilinear', mode='L')-.5
              for a in im]
        z = np.zeros((N, self.K))
        X = np.array(im).reshape(N, -1)
        return X, z, z, self.lmbd

    def lasso_cost(self, z, sig):
        '''Cost of the point z for a problem with sig'''
        residual = sig - z.dot(self.D)
        Er = np.sum(residual*residual, axis=1)/2
        return Er.mean() + self.lmbd*abs(z).sum(axis=1).mean()


def create_dictionary_dl(lmbd, K=100, N=10000):
    from sklearn.decomposition import DictionaryLearning
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    im = mnist.train.next_batch(N)[0]
    im = im.reshape(N, 28, 28)
    im = [imresize(a, (17, 17), interp='bilinear', mode='L')-.5
          for a in im]
    X = np.array(im).reshape(N, -1)
    print(X.shape)

    dl = DictionaryLearning(K, alpha=lmbd*N, fit_algorithm='cd',
                            n_jobs=-1, verbose=1)
    dl.fit(X)
    return dl.components_.reshape(K, -1)
