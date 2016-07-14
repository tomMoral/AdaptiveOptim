import numpy as np
import tensorflow as tf
from sys import stdout as out


class _OptimTF(object):
    """Iterative Soft thresholding algorithm in TF"""
    def __init__(self, name, gpu_usage=1):
        super().__init__()
        self.name = name
        self.gpu_usage = gpu_usage
        self._construct()
        self.reset()

    def _construct(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs = self._get_inputs()
            self.step_optim = self._step_optim(self.inputs)
            self._cost = self._get_cost(self.inputs)
            self.var_init = tf.initialize_all_variables()

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

    def _convert_feed(self, feed):
        _feed = {}
        for k, v in feed.items():
            if k in self.feed_map.keys():
                _feed[self.feed_map[k]] = v
        return _feed

    def optimize(self, X, lmbd, Z=None, max_iter=1, tol=1e-5):
        if Z is None:
            batch_size = X.shape[0]
            K = self.D.shape[0]
            Z = np.zeros((batch_size, K))

        z_ista = np.copy(Z)
        self.train_cost = []
        dE = 1
        feed = {self.X: X, self.Z: z_ista, self.lmbd: lmbd}
        for k in range(max_iter):
            z_ista[:], cost = self.session.run([self.step_optim, self._cost],
                                               feed_dict=feed)
            self.train_cost += [cost]
            if k > 0:
                dE = 1 - self.train_cost[-1]/self.train_cost[-2]
            if dE < tol:
                print("\r{} reached optimal solution in {}-iteration"
                      .format(self.name, k))
                break
            out.write("\rIterative optimization ({}): {:7.1%} - {:.4f}"
                      "".format(self.name, k/max_iter, dE))
            out.flush()
        self.train_cost += [self.session.run(self._cost, feed_dict=feed)]
        print("\rIterative optimization ({}): {:7}".format(self.name, "done"))
        return z_ista

    def _convert_feed(self, feed):
        _feed = {}
        for k, v in feed.items():
            _feed[self.feed_map[k]] = v
        return _feed
