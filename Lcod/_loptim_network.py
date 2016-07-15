import numpy as np
import os.path as osp
import tensorflow as tf
from sys import stdout as out


class _LOptimNetwork(object):
    """Lista Neural Network"""
    def __init__(self, n_layers, shared=False, warm_param=[], name='Loptim',
                 gpu_usage=1):
        self.n_layers = n_layers
        self.shared = shared
        self.warm_param = warm_param
        self.gpu_usage = gpu_usage
        self.name = name

        self._construct()
        self.reset()

    def _construct(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope(self.name):
                # Declare the training variables
                self.lr = tf.placeholder(dtype=tf.float32,
                                         name='learning_rate')
                self.global_step = tf.Variable(
                    tf.constant(0, dtype=tf.float32), name="step")

                # Construct the first layer from the inputs of the network
                inputs = self._get_inputs()
                outputs, params = self._layer(inputs, id_layer=0)
                self.param_layers = [params]
                if not self.shared:
                    params = None

                # Construct the next layers by chaining the outputs
                for k in range(self.n_layers-1):
                    outputs, params = self._layer(outputs, params=params,
                                                  id_layer=k+1)
                    if not self.shared:
                        self.param_layers += [params]
                        params = None

                # Construct and store the output/cost operation for the network
                self._output = self._get_output(outputs)
                self._cost = self._get_cost(outputs)
                c_val = tf.constant(0, dtype=tf.float32, name='c_val')
                c_tst = tf.constant(0, dtype=tf.float32, name='c_tst')
                tf.scalar_summary('training/cost', tf.log(self._cost-c_val))
                tf.scalar_summary('training/learning_rate', self.lr)
                self.tst_cost = tf.scalar_summary(
                    'testing/cost', tf.log(self._cost-c_tst),
                    collections=['TEST'])
                self.feed_map['c_val'], self.feed_map['c_tst'] = c_val, c_tst

                # Training methods
                with tf.name_scope('Training'):
                    self._optimizer = tf.train.AdagradOptimizer(self.lr)
                    grads_and_vars = self._optimizer.compute_gradients(
                        self._cost)
                    # for g, v in grads_and_vars:
                    #     if g is not None:
                    #         tf.histogram_summary(
                    #             osp.basename(v.name)+'_grad',
                    #             tf.log(self.lr*tf.clip_by_value(tf.abs(g),
                    #                    1e-7, 1e8))/np.log(10))
                    #         tf.histogram_summary(
                    #             osp.basename(v.name)+'_var', v)
                    self._train = self._optimizer.apply_gradients(
                        grads_and_vars)
                    self._inc = self.global_step.assign_add(1)

            self.var_init = tf.initialize_all_variables()
            self.saver = tf.train.Saver([pl for pp in self.param_layers
                                         for pl in pp] + [self.global_step])

            self.summary = tf.merge_all_summaries()
            import shutil
            if osp.exists('/tmp/TensorBoard/'+self.name):
                shutil.rmtree('/tmp/TensorBoard/'+self.name)
            else:
                if not osp.exists('/tmp/TensorBoard'):
                    import os
                    os.mkdir('/tmp/TensorBoard')
            self.writer = tf.train.SummaryWriter('/tmp/TensorBoard/'+self.name,
                                                 self.graph)

    def _layer(self, input, params=None, id_layer=0):
        '''Construct a layer from input and return its output

        This method takes the output of the previous layer of the network as
        its input and can reuse some parameter of the previous layer thru the
        params keyword.
        It should return a tuple composed of the outputs and the parameters of
        the layer. The outputs should
        '''
        raise NotImplementedError("{} must implement the _layer method"
                                  "".format(self.__class__))

    def _get_inputs(self):
        '''Construct the placeholder used for the network inputs
        '''
        raise NotImplementedError("{} must implement the _get_inputs method"
                                  "".format(self.__class__))

    def _get_output(self, outputs):
        '''Select the output of the network from the outputs of the last layer
        '''
        return outputs

    def _get_feed(self, batch_provider):
        '''Construct the feed dictionary from the batch provider

        This method will be use to feed the network at each step of the
        optimization from the batch provider. It will put in correspondance
        the tuple return by the batch_provider and the input placeholders.
        '''
        raise NotImplementedError("{} must implement the _get_feed method"
                                  "".format(self.__class__))

    def _get_cost(self, outputs):
        '''Construct the cost function from the outputs of the last layer'''
        raise NotImplementedError("{} must implement the _get_cost method"
                                  "".format(self.__class__))

    def reset(self):
        '''Reset the state of the network.'''
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_usage
        self.session = tf.Session(graph=self.graph, config=config)
        self.train_cost = []
        self.test_cost = []
        self.session.run(self.var_init)

    def terminate(self):
        self.session.close()

    def restore(self, model_name='loptim'):
        self.saver.restore(self.session, "save_exp/{}-{}.ckpt"
                           "".format(model_name, self.n_layers))

    def train(self, batch_provider, max_iter, steps, feed_test, feed_val,
              lr_init=.01, tol=1e-5, reg_cost=15, model_name='loptim',
              save_model=False, test_cost=False):
        feed_test = self._convert_feed(feed_test)
        feed_val = self._convert_feed(feed_val)
        dE = 1
        mE, params = 1e10, self.export_param()
        training_cost = [1e10]
        ds = -reg_cost
        with self.session.as_default():
            for k in range(max_iter*steps):
                it = self.global_step.eval()
                if it % steps == 0:
                    feed_val[self.lr] = lr_init*np.log(np.e+it/100)
                    cost, summary = self.session.run(
                        [self._cost, self.summary], feed_dict=feed_val)
                    self.train_cost += [cost]
                    self.writer.add_summary(summary, it)
                    if test_cost:
                        feed_test[self.lr] = lr_init*np.log(np.e+it/100)
                        cost, summary = self.session.run(
                            [self._cost, self.tst_cost], feed_dict=feed_test)
                        self.test_cost += [cost]
                        self.writer.add_summary(summary, it)
                    if mE > self.train_cost[-1]:
                        params = self.export_param()
                        mE = self.train_cost[-1]

                    if len(self.train_cost) > 2*reg_cost:
                        dE = (1 - np.mean(self.train_cost[-reg_cost:]) /
                              np.mean(self.train_cost[-2*reg_cost:-reg_cost]))

                        if dE < tol and (it - ds) >= (reg_cost // 2):
                            print("\rDownscale lr at iteration {} - ({:10.3e})"
                                  .format(it, dE))
                            lr_init *= .95
                            ds = it
                            if lr_init < 1e-5:
                                print("Learning rate too low, stop")
                                break

                out.write("\rTraining {}: {:7.2%} - {:10.3e}"
                          .format(self.name, k/(max_iter*steps), dE))
                out.flush()
                feed_dict = self._get_feed(batch_provider)
                feed_dict[self.lr] = lr_init*np.log(np.e+it/100)
                cost, _, _ = self.session.run(
                    [self._cost, self._train, self._inc], feed_dict=feed_dict)
                training_cost += [cost]
                if len(training_cost) > 16 and (
                        1.1*np.mean(training_cost[-16:-8]) <
                        np.mean(training_cost[-8:])):
                    self.import_param(params)
                    print('\rDownscale lr', training_cost[-2:])
                    training_cost.pop(-1)
                    lr_init *= .96

            self.import_param(params)
            self.train_cost += [self._cost.eval(feed_dict=feed_val)]
            self.test_cost += [self._cost.eval(feed_dict=feed_test)]
            self.writer.flush()
            print("\rTraining {}: {:7}".format(self.name, "done"))

            # Save the variables to disk.
            if save_model:
                save_path = self.saver.save(
                    self.session,
                    "save_exp/{}-{}.ckpt".format(model_name, self.n_layers),
                    global_step=self.global_step)
                print("Model saved in file: %s" % save_path)

    def output(self, **feed_dict):
        feed_dict = self._convert_feed(feed_dict)
        with self.session.as_default():
            return self._output.eval(feed_dict=feed_dict)

    def cost(self, **feed_dict):
        feed_dict = self._convert_feed(feed_dict)
        return self.session.run(self._cost, feed_dict=feed_dict)

    def _convert_feed(self, feed):
        _feed = {}
        for k, v in feed.items():
            if k in self.feed_map.keys():
                _feed[self.feed_map[k]] = v
        return _feed

    def export_param(self, n_layer=None):
        export = []
        with self.session.as_default():
            for params in self.param_layers[:n_layer]:
                pl = []
                for p in params:
                    pl += [p.eval()]
                export += [pl]
        return export

    def import_param(self, wp, n_layer=None):
        to_run = []
        with self.session.as_default():
            with self.graph.as_default():
                for wpl, params in zip(wp, self.param_layers[:n_layer]):
                    for w, p in zip(wpl, params):
                        to_run += [p.assign(tf.constant(w))]

            self.session.run(to_run)
            w, vp = wp[0][0], self.param_layers[0][0].eval()
            assert np.allclose(w, vp), "Issue in import param!!"
