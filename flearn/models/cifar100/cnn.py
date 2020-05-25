import numpy as np
import tensorflow as tf
from tqdm import trange

from flearn.utils.model_utils import batch_data, suffer_data, get_random_batch_sample
from flearn.utils.tf_utils import graph_size
from flearn.utils.tf_utils import process_grad, prox_L2


class Model(object):
    '''
    Assumes that images are 32px by 32px by 3 (32x32x3)
    '''
    
    def __init__(self, num_classes, optimizer, seed=1):

        # params
        # print("**** Numb classes:", num_classes)
        self.num_classes = num_classes
        self.optimizer = optimizer
        #self.create_model(optimizer)
        # create computation graph        
        self.graph = tf.Graph()

        with self.graph.as_default():
            tf.set_random_seed(123+seed)
            self.features, self.labels, self.train_op, self.grads, self.eval_metric_ops, self.loss = self.create_model(optimizer)
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph)

        # find memory footprint and compute cost of the model
        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops
    
    def create_model(self, optimizer):
        """Model function for Logistic Regression."""
        features = tf.placeholder(tf.float32, shape=[None, 3072], name='features')
        labels = tf.placeholder(tf.int64, shape=[None,], name='labels')
        input_layer = tf.reshape(features, [-1, 32, 32, 3]) #32 x 32 x3  = 3072=> Cifar100
        conv1 = tf.layers.conv2d(inputs=input_layer, filters=32,  kernel_size=[5, 5], padding="same",  activation=tf.nn.relu)
        print("conv1:",conv1.shape)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        print("pool1:",pool1.shape)
        dropout1 = tf.layers.dropout(
            pool1,
            rate=0.4,
        )
        print("dropout1:",dropout1.shape)
        conv2 = tf.layers.conv2d(
            inputs=dropout1,
            filters=64,
            # filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        print("conv2:", conv2.shape)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        print("pool2:", pool2.shape)
        dropout2 = tf.layers.dropout(
            pool2,
            rate=0.4,
        )
        print("dropout2:", dropout2.shape)
        # pool2_flat = tf.reshape(dropout2, [-1, 8 * 8 * 32])  # 7 * 7 * 64
        pool2_flat = tf.reshape(dropout2, [-1, 8 * 8 * 64])  #7 * 7 * 64
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        # dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=dense, units=self.num_classes)
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        return features, labels, train_op, grads, eval_metric_ops, loss

    def set_params(self, model_params=None):
        if model_params is not None:
            with self.graph.as_default():
                all_vars = tf.trainable_variables()
                for variable, value in zip(all_vars, model_params):
                    variable.load(value, self.sess)
    def set_vzero(self, vzero):
        self.vzero = vzero

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    def get_gradients(self, data, model_len):

        grads = np.zeros(model_len)
        num_samples = len(data['y'])

        with self.graph.as_default():
            model_grads = self.sess.run(self.grads,
                feed_dict={self.features: data['x'], self.labels: data['y']})
            grads = process_grad(model_grads)

        return num_samples, grads
    
    def get_raw_gradients(self, data):
    
        with self.graph.as_default():
            model_grads = self.sess.run(self.grads,
                                        feed_dict={self.features: data['x'], self.labels: data['y']})

        return model_grads

    def set_gradientParam(self, preG, preGn):
        self.optimizer.set_preG(preG, self)
        self.optimizer.set_preGn(preGn, self)

    def solve_inner(self, optimizer, data, num_epochs=1, batch_size=32):
        '''Solves local optimization problem'''
        if (batch_size == 0):  # Full data or batch_size
            batch_size = len(data['y'])  # //10

        #if(optimizer == "fedavg"):
        #data_x, data_y = suffer_data(data)
        for _ in trange(num_epochs, desc='Epoch: ', leave=False, ncols=120):
            #X, y = get_random_batch_sample(data_x, data_y, batch_size)
            #with self.graph.as_default():
            #    self.sess.run(self.train_op, feed_dict={self.features: X, self.labels: y})
            for X, y in batch_data(data, batch_size):
                with self.graph.as_default():
                    self.sess.run(self.train_op, feed_dict={
                                  self.features: X, self.labels: y})
        soln = self.get_params()
        with self.graph.as_default():
            grad = self.sess.run(self.grads, feed_dict={
                                 self.features: data['x'], self.labels: data['y']})
        comp = num_epochs * \
            (len(data['y'])//batch_size) * batch_size * self.flops
        return soln, grad, comp
    
    def test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        with self.graph.as_default():
            # print(data['x'].shape)
            # print(data['y'].shape)
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss], 
                feed_dict={self.features: data['x'], self.labels: data['y']})
        return tot_correct, loss
    
    def close(self):
        self.sess.close()