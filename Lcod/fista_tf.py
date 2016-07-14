import numpy as np
import tensorflow as tf
from sys import stdout as out

from .helper_tf import soft_thresholding
from ._optim_tf import _OptimTF


class FistaTF(_OptimTF):
    """Iterative Soft thresholding algorithm in TF"""
    def __init__(self, D):
        self.D = np.array(D).astype(np.float32)
        self.S0 = D.dot(D.T).astype(np.float32)
        self.L = np.linalg.norm(D, ord=2)**2

        super().__init__(name='FISTA')

    def _get_inputs(self):
        K, p = self.D.shape
        self.Z = tf.placeholder(shape=[None, K], dtype=tf.float32, name='Z')
        self.Y = tf.placeholder(shape=[None, K], dtype=tf.float32, name='Yk')
        self.X = tf.placeholder(shape=[None, p], dtype=tf.float32,  name='X')
        self.theta = tf.placeholder(dtype=tf.float32, name='theta')
        self.lmbd = tf.placeholder(dtype=tf.float32, name='lmbd')
        self.feed_map = {"Z": self.Z, "X": self.X, "theta": self.theta,
                         "lmbd": self.lmbd}

        return (self.Z, self.Y, self.X, self.theta, self.lmbd)

    def _step_optim(self, inputs):
        Z, Y, X, theta, lmbd = self.inputs
        K, p = self.D.shape
        L = self.L
        with tf.name_scope("ISTA_iteration"):
            self.S = tf.constant(np.eye(K, dtype=np.float32) - self.S0/L,
                                 shape=[K, K], name='S')
            self.We = tf.constant(self.D.T/L, shape=[p, K],
                                  dtype=tf.float32, name='We')
            B = tf.matmul(X, self.We, name='B')
            hk = tf.matmul(Y, self.S) + B
            self.step_FISTA = Zk = soft_thresholding(hk, lmbd/L)
            self.theta_k = tk = (tf.sqrt(theta**4 + 4*theta*theta) -
                                 theta*theta)/2
            self.Yk = Zk + tk*(1/theta-1)*tf.sub(Zk, Z)

            step = tf.tuple([Zk, tk, self.Yk])
        return step

    def _get_cost(self, inputs):
        Z, _, X, _, lmbd = self.inputs
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

    def optim(self, X, lmbd, Z=None, max_iter=1, tol=1e-5):
        if Z is None:
            batch_size = X.shape[0]
            K = self.D.shape[0]
            Z = np.zeros((batch_size, K))

        z_fista = np.copy(Z)
        yk = np.copy(z_fista)
        tk = 1
        dE = 1
        self.train_cost = []
        for k in range(max_iter):
            feed = {self.X: X, self.Z: z_fista, self.Y: yk, self.theta: tk,
                    self.lmbd: lmbd}
            z_fista[:], yk[:], tk, cost = self.session.run([
                self.step_FISTA, self.Yk, self.theta_k, self._cost
                ], feed_dict=feed)
            self.train_cost += [cost]
            if k > 0:
                dE = 1 - self.train_cost[-1]/self.train_cost[-2]
            if dE < tol:
                print("\r{} reached optimal solution in {}-iteration"
                      .format(self.name, k))
                break
            out.write("\rIterative optimization (FISTA): {:7.1%} - {:.4f}"
                      .format(k/max_iter, dE))
            out.flush()
        print("\rIterative optimization (FISTA): {:7}".format("done"))

    def _convert_feed(self, feed):
        _feed = {}
        for k, v in feed.items():
            _feed[self.feed_map[k]] = v
        return _feed
