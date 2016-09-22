import logging
import numpy as np
import tensorflow as tf

from .utils import start_handler
from .helper_tf import soft_thresholding
from ._loptim_network import _LOptimNetwork


class LFistaNetwork(_LOptimNetwork):
    """Lifsta Neural Network"""
    def __init__(self, D, n_layers, log_lvl=logging.INFO, name=None, **kwargs):
        self.D = np.array(D).astype(np.float32)
        self.S0 = D.dot(D.T).astype(np.float32)
        self.L = np.linalg.norm(D, ord=2)**2

        tk, theta_k = 1, [1]
        for _ in range(n_layers+1):
            tk = (np.sqrt((tk*tk+4)*tk*tk) - tk*tk)/2
            theta_k += [tk]
        self.theta_k = theta_k

        self.log = logging.getLogger('LFistaNet')
        start_handler(self.log, log_lvl)

        if name is None:
            name = 'L-FISTA_{:03}'.format(n_layers)

        super().__init__(n_layers=n_layers, name=name, **kwargs)

    def _get_inputs(self):
        """Construct the placeholders used for the network inputs, to be passed
        as entries for the first layer.

        Return
        ------
        outputs: tuple of tensors (n_in) passed as entries to construct the 1st
                 layer of the network.
        """
        K, p = self.D.shape
        self.X = tf.placeholder(shape=[None, p], dtype=tf.float32,
                                name='signal')
        self.lmbd = tf.placeholder(dtype=tf.float32, name='lambda')
        self.Z = tf.zeros(shape=[tf.shape(self.X)[0], K], dtype=tf.float32,
                          name='Z_0')

        self.feed_map = {"Z": self.Z, "X": self.X, "lmbd": self.lmbd}
        return (self.Z, tf.zeros_like(self.Z, name='Y_0'), self.X, self.lmbd)

    def _get_cost(self, outputs):
        """Construct the cost function from the outputs of the last layer. This
        will be used through SGD to train the network.

        Parameters
        ----------
        outputs: tuple fo tensors (n_out)
            a tuple of tensor containing the output from the last layer of the
            network

        Returns
        -------
        cost: a tensor computing the cost function of the network.
        reg: a tensor for computing regularisation of the parameters.
            It should be 0 if no regularization is needed.
        """
        Zk, _, X, lmbd = outputs

        with tf.name_scope("reconstruction_zD"):
            rec = tf.matmul(Zk, tf.constant(self.D))

        with tf.name_scope("norm_2"):
            Er = .5*tf.reduce_mean(tf.reduce_sum(
                tf.squared_difference(rec, X), reduction_indices=[1]))

        with tf.name_scope("norm_1"):
            l1 = lmbd*tf.reduce_mean(tf.reduce_sum(
                tf.abs(Zk), reduction_indices=[1]))

        return Er + l1

    def _get_feed(self, batch_provider):
        """Construct the feed dictionary from the batch provider

        This method will be use to feed the network at each step of the
        optimization from the batch provider. It will put in correspondance
        the tuple return by the batch_provider and the input placeholders.
        """
        sig_batch, _, zs_batch, lmbd = batch_provider.get_batch()
        feed_dict = {self.Z: zs_batch,
                     self.X: sig_batch,
                     self.lmbd: lmbd}
        return feed_dict

    def _layer(self, inputs, params=None, id_layer=0):
        """Construct the layer id_layer in the computation graph of tensorflow.

        Parameters
        ----------
        inputs: tuple of tensors (n_in)
            a tuple of tensor containing all the necessary inputs to construct
            the layer, either network inputs or previous layer output.
        params: tuple of tensor (n_param)
            a tuple with the parameter of the previous layers, used to share
            the parameters accross layers. This is not used if the network do
            not use the shared parameter.
        id_layer: int
            A layer identifier passed during the construction of the network.
            It should be its rank in the graph.
        Returns
        -------
        outputs: tuple of tensors (n_out) st n_out = n_in, to chain the layers.
        params: tuple of tensors (n_param) with the parameters of this layer
        """
        Z, Y, X, lmbd = inputs
        L, (K, p), k = self.L, self.D.shape, id_layer
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

            if id_layer > 0 or self.shared:
                Wg = tf.Variable(tf.constant(wp[0], shape=[K, K]), name='Wg')
                Wm = tf.Variable(tf.constant(wp[1], shape=[K, K]), name='Wm')
            else:
                Wg = Wm = None
            We = tf.Variable(tf.constant(wp[2], shape=[p, K]),  name='We')
            theta = tf.Variable(tf.constant(wp[3], shape=[K]), name='theta')

        with tf.name_scope("hidden_{}".format(id_layer)):
            hk = tf.matmul(self.X, We)
            if id_layer > 0:
                hk += tf.matmul(Y, Wg)
        Zk = soft_thresholding(hk, self.lmbd*theta)
        if id_layer > 0:
            Yk = Zk + tf.matmul(tf.sub(Zk, Z), Wm)
        else:
            Yk = Zk

        tf.identity(Zk, name="output")

        return (Zk, Yk, X, lmbd), (Wg, Wm, We, theta)
