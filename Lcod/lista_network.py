import logging
import numpy as np
import tensorflow as tf
from sys import stdout as out

from .helper_tf import soft_thresholding
from ._loptim_network import _LOptimNetwork


class LIstaNetwork(_LOptimNetwork):
    """Lifsta Neural Network"""
    def __init__(self, D, n_layers, shared=False, warm_param=[],
                 lin_layer=False, log_lvl=logging.INFO, gpu_usage=1):
        self.D = np.array(D).astype(np.float32)
        self.S0 = D.dot(D.T).astype(np.float32)
        self.L = np.linalg.norm(D, ord=2)**2
        self.log = logging.getLogger('LIstaNet')
        self.lin_layer = lin_layer
        if len(self.log.handlers) == 0:
            ch = logging.StreamHandler(out)
            ch.setLevel(log_lvl)
            formatter = logging.Formatter('\r[%(name)s] - %(message)s')
            ch.setFormatter(formatter)
            self.log.addHandler(ch)

        super().__init__(n_layers=n_layers, shared=shared,
                         warm_param=warm_param, gpu_usage=gpu_usage,
                         name='L-ISTA_{:03}'.format(n_layers))

    def _get_inputs(self):
        K, p = self.D.shape
        # with tf.name_scope("Inputs"):
        self.X = tf.placeholder(shape=[None, p], dtype=tf.float32,
                                name='X')
        self.Z = tf.placeholder_with_default(
            tf.zeros(shape=[tf.shape(self.X)[0], K], dtype=tf.float32),
            shape=[None, K], name='Z_0')
        self.lmbd = tf.placeholder(dtype=tf.float32, name='lambda')

        self.feed_map = {"Z": self.Z, "X": self.X, "lmbd": self.lmbd}
        return [self.Z, self.X, self.lmbd]

    def _get_output(self, outputs):
        # K, _ = self.D.shape
        Wg, We, _ = self.param_layers[-1]
        if self.lin_layer:
            with tf.name_scope("Output"):
                with tf.name_scope("A"):
                    self.A = Wg + tf.matmul(self.D, We)
                Zk = tf.matmul(outputs[0], self.A)
            outputs[0] = Zk
        return outputs[0]

    def _get_cost(self, outputs):
        K, p = self.D.shape
        Zk, X, lmbd = outputs
        with tf.name_scope("Cost"):
            with tf.name_scope("reconstruction_zD"):
                rec = tf.matmul(Zk, tf.constant(self.D))
            with tf.name_scope("norm_2"):
                Er = tf.reduce_mean(
                    tf.reduce_sum(tf.squared_difference(rec, X),
                                  reduction_indices=[1]))/2
            with tf.name_scope("norm_1"):
                l1 = lmbd*tf.reduce_mean(
                    tf.reduce_sum(tf.abs(Zk), reduction_indices=[1]),
                    name='norm1')

            cost = Er + l1
        return cost

    def _get_feed(self, batch_provider):
        sig_batch, _, zs_batch, lmbd = batch_provider.get_batch()
        feed_dict = {self.Z: zs_batch,
                     self.X: sig_batch,
                     self.lmbd: lmbd}
        return feed_dict

    def _layer(self, inputs, params=None, id_layer=0):
        '''Construct a layer for the
        '''
        L = self.L
        K, p = self.D.shape
        Zk, X, lmbd = inputs
        with tf.name_scope("Layer_Lista_{}".format(id_layer)):
            if params:
                self.log.debug('(Layer{}) - shared params'.format(id_layer))
                Wg, We, theta = params
            else:
                if len(self.warm_param) > id_layer:
                    self.log.debug('(Layer{})- warm params'.format(id_layer))
                    wp = self.warm_param[id_layer]
                else:
                    self.log.debug('(Layer{}) - new params'.format(id_layer))
                    wp = [np.eye(K, dtype=np.float32) - self.S0/L,
                          (self.D.T).astype(np.float32)/L,
                          np.ones(K, dtype=np.float32)/L]
                Wg = tf.Variable(tf.constant(wp[0], shape=[K, K]),
                                 name='Wg_{}'.format(id_layer))
                We = tf.Variable(tf.constant(wp[1], shape=[p, K]),
                                 name='We_{}'.format(id_layer))
                theta = tf.Variable(tf.constant(wp[2], shape=[K]),
                                    name='theta_{}'.format(id_layer))

            with tf.name_scope("hidden_{}".format(id_layer)):
                B = tf.matmul(self.X, We, name='B_{}'.format(id_layer))
                hk = tf.matmul(Zk, Wg) + B
            output = soft_thresholding(hk, self.lmbd*theta)

            return [output, X, lmbd], (Wg, We, theta)
