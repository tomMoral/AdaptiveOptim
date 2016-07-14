import numpy as np
import tensorflow as tf
from sys import stdout as out

from .helper_tf import soft_thresholding
from ._optim_tf import _OptimTF


class IstaTF(_OptimTF):
    """Iterative Soft thresholding algorithm in TF"""
    def __init__(self, D):
        self.D = np.array(D).astype(np.float32)
        self.S0 = D.dot(D.T).astype(np.float32)
        self.L = np.linalg.norm(D, ord=2)**2

        super().__init__(name='ISTA')

    def _get_inputs(self):
        K, p = self.D.shape
        self.Z = tf.placeholder(shape=[None, K], dtype=tf.float32,
                                name='Z')
        self.X = tf.placeholder(shape=[None, p], dtype=tf.float32,
                                name='X')
        self.lmbd = tf.placeholder(dtype=tf.float32, name='lmbd')
        self.feed_map = {"Z": self.Z, "X": self.X, "lmbd": self.lmbd}

        return (self.Z, self.X, self.lmbd)

    def _step_optim(self, inputs):
        Z, X, lmbd = self.inputs
        K, p = self.D.shape
        L = self.L
        with tf.name_scope("step_ISTA"):
            self.S = tf.constant(np.eye(K, dtype=np.float32) - self.S0/L,
                                 shape=[K, K], name='S')
            self.We = tf.constant(self.D.T/L, shape=[p, K],
                                  dtype=tf.float32, name='We')
            B = tf.matmul(X, self.We, name='B')
            hk = tf.matmul(Z, self.S) + B
            step = soft_thresholding(hk, lmbd/L)
        return step

    def _get_cost(self, inputs):
        Z, X, lmbd = self.inputs
        with tf.name_scope("Cost"):
            rec = tf.matmul(Z, tf.constant(self.D))
            Er = tf.reduce_mean(
                tf.reduce_sum(tf.squared_difference(rec, X),
                              reduction_indices=[1]))/2
            cost = Er + lmbd*tf.reduce_mean(
                tf.reduce_sum(tf.abs(Z), reduction_indices=[1]))

        return cost


    def output(self, X, z_start=None):
        if z_start is None:
            batch_size = X.shape[0]
            K = self.D.shape[0]
            z_start = np.zeros((batch_size, K))

        feed = {self.X: X, self.Z: z_start}
        return self._output.eval(feed_dict=feed, session=self.session)

    # def optimize(self, X, lmbd, Z=None, max_iter=1, tol=1e-5):
    #     if Z is None:
    #         batch_size = X.shape[0]
    #         K = self.D.shape[0]
    #         Z = np.zeros((batch_size, K))

    #     z_ista = np.copy(Z)
    #     feed = {self.X: X, self.Z: z_ista, self.lmbd: lmbd}
    #     self.train_cost = []
    #     dE = 1
    #     feed = {self.X: X, self.Z: z_ista, self.lmbd: lmbd}
    #     for k in range(max_iter):
    #         z_ista[:], cost = self.session.run([self.step_ISTA, self._cost],
    #                                            feed_dict=feed)
    #         self.train_cost += [cost]
    #         if k > 0:
    #             dE = 1 - self.train_cost[-1]/self.train_cost[-2]
    #         if dE < tol:
    #             print("\rReach optimal solution in {}-iteration: {:.5e}"
    #                   "".format(k, dE))
    #             break
    #         out.write("\rOptim ISTA: {:7.1%} - {:.4f}"
    #                   "".format(k/max_iter, dE))
    #         out.flush()
    #     self.train_cost += [self.session.run(self._cost, feed_dict=feed)]
    #     print("\rOptim ISTA:: {:7}".format("done"))
    #     return z_ista

    def _convert_feed(self, feed):
        _feed = {}
        for k, v in feed.items():
            _feed[self.feed_map[k]] = v
        return _feed
