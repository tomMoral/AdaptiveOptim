import numpy as np
import tensorflow as tf

from .helper_tf import soft_thresholding
from ._loptim_network import _LOptimNetwork


class LFistaNetwork(_LOptimNetwork):
    """Lifsta Neural Network"""
    def __init__(self, D, n_layers, shared=False, warm_param=[], gpu_usage=1):
        self.D = np.array(D).astype(np.float32)
        self.S0 = D.dot(D.T).astype(np.float32)
        self.L = np.linalg.norm(D, ord=2)**2

        tk, theta_k = 1, [1]
        for _ in range(n_layers+1):
            tk = (np.sqrt((tk*tk+4)*tk*tk) - tk*tk)/2
            theta_k += [tk]
        self.theta_k = theta_k

        super().__init__(n_layers=n_layers, shared=shared,
                         warm_param=warm_param, gpu_usage=gpu_usage,
                         name='L-FISTA_{:03}'.format(n_layers))

    def _get_inputs(self):
        K, p = self.D.shape
        self.Z = tf.placeholder(shape=[None, K], dtype=tf.float32,
                                name='Z_0')
        self.X = tf.placeholder(shape=[None, p], dtype=tf.float32,
                                name='signal')
        self.lmbd = tf.placeholder(dtype=tf.float32, name='lambda')

        self.feed_map = {"Z": self.Z, "X": self.X, "lmbd": self.lmbd}
        return (self.Z, tf.zeros_like(self.Z, name='Y_0'), self.X, self.lmbd)

    def _get_output(self, outputs):
        return outputs[0]

    def _get_cost(self, outputs):
        Zk, _, X, lmbd = outputs
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

        cost = tf.add(Er, l1, name='cost')
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
        Z, Y, X, lmbd = inputs
        L, (K, p), k = self.L, self.D.shape, id_layer
        with tf.name_scope("Layer_Lfista_{}".format(id_layer)):
            if params:
                Wg, Wm, We, theta = params
            else:
                if len(self.warm_param) > id_layer:
                    wp = self.warm_param[id_layer]
                else:
                    mk = self.theta_k[k+1]*(1/self.theta_k[k]-1)
                    wp = [np.eye(K, dtype=np.float32) - self.S0/L,
                          mk*np.eye(K, dtype=np.float32),
                          (self.D.T/L).astype(np.float32),
                          np.ones(K, dtype=np.float32)/L]
                Wg = tf.Variable(tf.constant(wp[0], shape=[K, K]),
                                 name='Wg_{}'.format(id_layer))
                Wm = tf.Variable(tf.constant(wp[1], shape=[K, K]),
                                 name='Wm_{}'.format(id_layer))
                We = tf.Variable(tf.constant(wp[2], shape=[p, K]),
                                 name='We_{}'.format(id_layer))
                theta = tf.Variable(tf.constant(wp[3], shape=[K]),
                                    name='theta_{}'.format(id_layer))

            with tf.name_scope("hidden_{}".format(id_layer)):
                B = tf.matmul(self.X, We, name='B_{}'.format(id_layer))
                hk = tf.matmul(Y, Wg) + B
            Zk = soft_thresholding(hk, self.lmbd*theta)
            with tf.name_scope("Y_{}".format(id_layer)):
                Yk = Zk + tf.matmul(tf.sub(Zk, Z), Wm)

            return (Zk, Yk, X, lmbd), (Wg, Wm, We, theta)
