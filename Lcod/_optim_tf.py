import numpy as np
import tensorflow as tf
from sys import stdout as out


class _OptimTF(object):
    """Iterative Soft thresholding algorithm in TF"""
    def __init__(self, name, gpu_usage=.9):
        super().__init__()
        self.name = name
        self.gpu_usage = gpu_usage
        self._construct()
        self.reset()

    def _construct(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs = self._get_inputs()
            self.step_optim, self.dz = self._get_step(self.inputs)
            self._cost = self._get_cost(self.inputs)
            self.var_init = tf.global_variables_initializer()

    def reset(self):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_usage
        self.session = tf.Session(graph=self.graph, config=config)
        self.train_cost = []
        self.session.run(self.var_init)

    def output(self, **feed_dict):
        feed_dict = self._convert_feed(feed_dict)
        return self._output.eval(feed_dict=feed_dict, session=self.session)

    def cost(self, **feed_dict):
        feed_dict = self._convert_feed(feed_dict)
        return self.session.run(self._cost, feed_dict=feed_dict)

    def optimize(self, X, lmbd, Z=None, max_iter=1, tol=1e-5):
        if Z is None:
            batch_size = X.shape[0]
            K = self.D.shape[0]
            z_curr = np.zeros((batch_size, K))
        else:
            z_curr = np.copy(Z)
        self.train_cost, self.train_z = [], []
        feed = {self.X: X, self.Z: z_curr, self.lmbd: lmbd}
        for k in range(max_iter):
            z_curr[:], dz, cost = self.session.run(
                [self.step_optim, self.dz, self._cost], feed_dict=feed)
            self.train_cost += [cost]
            self.train_z += [np.copy(z_curr)]
            if dz < tol:
                print("\r{} reached optimal solution in {}-iteration"
                      .format(self.name, k))
                break
            out.write("\rIterative optimization ({}): {:7.1%} - {:.4e}"
                      "".format(self.name, k/max_iter, dz))
            out.flush()
        self.train_cost += [self.session.run(self._cost, feed_dict=feed)]
        print("\rIterative optimization ({}): {:7}".format(self.name, "done"))
        return z_curr

    def _convert_feed(self, feed):
        _feed = {}
        for k, v in feed.items():
            _feed[self.feed_map[k]] = v
        return _feed

    def terminate(self):
        self.session.close()
