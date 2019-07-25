'''
-- Difference to mytcn.py: automatic scaling of noise --
Is used for ICANN 2019 paper.

Builds a complete Temporal Convolution Network (with output layer, train and 
test functions). 

Main parts of code taken from add_test.py from: 
https://github.com/YuanTingHsieh/TF_TCN/blob/master/adding_problem/add_test.py

Modified code to enable Monte Carlo dropout: 
    - placeholder for dropout (now dropout is also possible during testing)
    - separate training and loss function for observation noise

Created on Jan 11, 2019

@author: ameier
'''

from pathlib import Path
import time
import warnings

from sklearn.metrics.pairwise import paired_distances
from tensorflow.python.ops.distributions.special_math import erfinv

import numpy as np
from predictors.tcn import TemporalConvNet
import tensorflow as tf


print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())


class MyAutoTCN():

    def __init__(self, in_channels, output_size, num_channels, sequence_length,
                 kernel_size, batch_size, n_train_mc_runs,
                 train_dropout, test_dropout, learning_rate, init=False, use_uncs=False):
        '''
        @param default_dropout: is overwritten in train() and evaluate()
        @param use_uncs: True if the loss function for aleatoric uncertainty 
        should be applied and epistemic uncertainties are computed
        '''
        self.lr = learning_rate
        self.batch_size = batch_size

        # number of Monte Carlo runs during training (for loss function with
        # automatic noise scaling)
        self.n_train_mc_runs = n_train_mc_runs
        self.train_dropout = train_dropout
        self.test_dropout = test_dropout
        self.use_uncs = use_uncs
        n_classes = output_size

        # None instead of batch_size
        self.input_pl = tf.placeholder(
            tf.float32, (None, sequence_length, in_channels))
        # None instead of batch_size
        self.output_pl = tf.placeholder(
            tf.float32, (None, n_classes))
        # default value for dropout (is for testing/training)
        self.dropout_pl = tf.placeholder_with_default(self.train_dropout, ())

        tcn = TemporalConvNet(input_layer=self.input_pl, num_channels=num_channels, sequence_length=sequence_length,
                              kernel_size=kernel_size, dropout=self.dropout_pl, init=init)

        # output layer for prediction
        self.out_layer = tf.contrib.layers.fully_connected(tcn[:, -1, :], output_size, activation_fn=None,
                                                           weights_initializer=tf.initializers.random_normal(0, 0.01))

        # deviation of prediction and labels
        prediction_mse = tf.losses.mean_squared_error(
            labels=self.output_pl, predictions=self.out_layer)
        self.pred_loss = prediction_mse

        # output layer for observation noise
        if self.use_uncs:
            self.noise_scope = "noise_output"
            # one neuron because one uncertainty output for one input
            self.n_neurons_aleat_unc = 1
            # for each dimension one uncertainty output
            self.n_neurons_aleat_unc = output_size
            # heteroscedastic aleatoric uncertainty
            with tf.variable_scope(self.noise_scope):
                self.noise_out_layer = tf.contrib.layers.fully_connected(tcn[:, -1, :], self.n_neurons_aleat_unc, activation_fn=tf.nn.softplus,
                                                                         weights_initializer=tf.initializers.random_normal(
                                                                         0, 0.01))
            # format [batch_size, dims]
            self.pred_mean_pl = tf.placeholder(tf.float32, (None, n_classes))
            # format [batch_size, dims]
            self.pred_var_pl = tf.placeholder(tf.float32, (None, n_classes))

            # format  [n_mc_runs-1, batch_size, dims] (n_mc_runs -1 since the
            # results for last run are added later)
            self.preds_pl = tf.placeholder(
                tf.float32, (self.n_train_mc_runs - 1, None, n_classes))
            # format  [n_mc_runs, batch_size, dims]
            self.al_uncs_pl = tf.placeholder(
                tf.float32, (self.n_train_mc_runs - 1, None, n_classes))

            self.pred_mean_pl, self.pred_var_pl = self.get_pred_var_and_mean(
                self.preds_pl, self.al_uncs_pl, self.out_layer, self.noise_out_layer)
            # compute loss
            t1_perc = 51 / 100
            t3_perc = 99 / 100

            t1 = self.diff(t1_perc) * (t1_perc - self.acc(t1_perc))**2
            t1 = tf.maximum(t1, 0)

            t2 = np.sum([abs(self.diff(i / 100) * (i / 100 - self.acc(i / 100))**2)
                         for i in range(52, 98 + 1)])

            t3 = self.diff(t3_perc) * (t3_perc - self.acc(t3_perc))**2
            t3 = tf.maximum(-t3, 0)

            self.noise_loss = 2 * t1 + t2 + 2 * t3

            # optimization operation for noise layer
            noise_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            # variables of noise layer
            noise_layer_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                 scope=self.noise_scope)
            # update step of noise layer (trains only that layer)
            self.noise_update_step = noise_optimizer.minimize(
                self.noise_loss, var_list=noise_layer_vars)
        else:
            self.noise_out_layer = []
            self.pred_loss = prediction_mse

        # print-output for prediction training
        tf.summary.scalar('mse', self.pred_loss)
        self.pred_merged = tf.summary.merge_all()

        # optimization operation for prediction layer
        pred_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        # update steps for prediction layer (trains all layers)
        self.pred_update_step = pred_optimizer.minimize(self.pred_loss)

    def index_generator(self, n_train):
        all_indices = np.arange(n_train)
        start_pos = 0
        # while True:
        #    all_indices = np.random.permutation(all_indices)
        for batch_idx, batch in enumerate(range(start_pos, n_train, self.batch_size)):

            start_ind = batch
            end_ind = start_ind + self.batch_size

            # last batch
            if end_ind > n_train:
                diff = end_ind - n_train
                toreturn = all_indices[start_ind:end_ind]
                toreturn = np.append(toreturn, all_indices[0:diff])
                yield batch_idx + 1, toreturn
                start_pos = diff
                break

            yield batch_idx + 1, all_indices[start_ind:end_ind]

    def shuffle_data(self, X_train, Y_train, n_train, pred_np_rnd_generator):
        # generate and shuffle indices to shuffle input and output data
        # in same order
        idx = np.arange(n_train)
        pred_np_rnd_generator.shuffle(idx)
        X_train = X_train[idx]
        Y_train = Y_train[idx]
        return X_train, Y_train

    def train(self, n_epochs, sess, X_train, Y_train, n_train, log_interval,
              train_writer, shuffle_between_epochs, pred_np_rnd_generator):
        # train output layer
        for ep in range(1, n_epochs + 1):
            if shuffle_between_epochs:
                X_train, Y_train = self.shuffle_data(
                    X_train, Y_train, n_train, pred_np_rnd_generator)
            self.train_prediction(ep, sess, X_train, Y_train,
                                  n_train, log_interval, train_writer)

        if self.use_uncs:
            # train noise layer
            for ep in range(1, n_epochs + 1):
                if shuffle_between_epochs:
                    X_train, Y_train = self.shuffle_data(
                        X_train, Y_train, n_train, pred_np_rnd_generator)
                self.train_noise(ep, sess, X_train, Y_train,
                                 n_train, log_interval, train_writer)

    def train_prediction(self, epoch, sess, X_train, Y_train, n_train, log_interval, train_writer):
        steps = 0
        total_loss = 0
        start_time = time.time()
        for batch_idx, indices in self.index_generator(n_train):
            x = X_train[indices]
            y = Y_train[indices]

            summary, _, p, l = sess.run([self.pred_merged, self.pred_update_step,
                                         self.out_layer, self.pred_loss],
                                        feed_dict={self.input_pl: x, self.output_pl: y,
                                                   self.dropout_pl: self.train_dropout})
            total_loss += l
            steps += 1
            if train_writer is not None:
                train_writer.add_summary(summary, steps)

            if (batch_idx > 0 and batch_idx % log_interval == 0):
                avg_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| Epoch {:3d} | {:5d}/{:5d} batches | lr {:2.5f} | ms/batch {:5.2f} | '
                      'loss {:5.8f} |'.format(
                          epoch, batch_idx, n_train // self.batch_size +
                          1, self.lr, elapsed * 1000 / log_interval,
                          avg_loss), flush=True)
                start_time = time.time()
                total_loss = 0

    def train_noise(self, epoch, sess, X_train, Y_train, n_train, log_interval, train_writer):
        if not self.use_uncs:
            warnings.warn("noise can not be trained since not activated!")

        print("\n train noise", flush=True)
        for batch_idx, indices in self.index_generator(n_train):
            x = X_train[indices]
            y = Y_train[indices]

            # format [n_mc_runs, batch_size, dims]
            preds = []
            al_uncs = []
            for _ in range(self.n_train_mc_runs - 1):
                p, al_unc = sess.run([self.out_layer, self.noise_out_layer], feed_dict={self.input_pl: x,
                                                                                        self.dropout_pl: self.train_dropout})
                preds.append(p)
                al_uncs.append(al_unc)

            # do weight update in last MC run
            _, _, _, l = sess.run([self.noise_update_step, self.out_layer,
                                   self.noise_out_layer, self.noise_loss],
                                  feed_dict={self.input_pl: x, self.output_pl: y,
                                             self.dropout_pl: self.train_dropout,
                                             self.preds_pl: preds,
                                             self.al_uncs_pl: al_uncs})

    def evaluate(self, sess, X_test, Y_test, n_test):

        total_pred = np.zeros(Y_test.shape)
        if self.n_neurons_aleat_unc > 1:
            total_aleat_unc_pred = np.zeros(Y_test.shape)
        else:
            total_aleat_unc_pred = np.zeros(len(Y_test))
        total_loss = []
        for batch_idx, batch in enumerate(range(0, n_test, self.batch_size)):
            start_idx = batch
            end_idx = batch + self.batch_size

            x = X_test[start_idx:end_idx]
            y = Y_test[start_idx:end_idx]

            exclude = 0
            if len(x) < self.batch_size:
                exclude = self.batch_size - len(x)
                x = np.pad(x, ((0, exclude), (0, 0), (0, 0)), 'constant')
                y = np.pad(y, ((0, exclude), (0, 0)), 'constant')

            p, al_un, l = sess.run([self.out_layer, self.noise_out_layer, self.pred_loss], feed_dict={
                self.input_pl: x, self.output_pl: y,
                self.dropout_pl: self.test_dropout})

            if self.n_neurons_aleat_unc <= 1:
                al_un = al_un.flatten()

            if exclude > 0:
                total_pred[start_idx:end_idx] = p[:-exclude]
                #total_loss += l
                total_aleat_unc_pred[start_idx:end_idx] = al_un[:-exclude]
            else:
                total_pred[start_idx:end_idx] = p
                total_loss.append(l)
                total_aleat_unc_pred[start_idx:end_idx] = al_un

        mse = np.mean(np.square(total_pred - Y_test))
        print('Test MSE Loss {:5.8f} '.format(mse))

        if not self.use_uncs:
            total_aleat_unc_pred = None

        distances = paired_distances(Y_test, total_pred)
        return total_pred, total_aleat_unc_pred, distances

    def predict(self, sess, X_test, n_test, n_features):
        '''
        Only prediction, no computation of MSE.
        (That is the difference to the function evaluate().)

        @param X_test: 3d array, format [n_data, n_time_steps, n_features]
        '''
        total_pred = np.zeros((n_test, n_features))
        if not self.use_uncs:
            pass
        else:
            if self.n_neurons_aleat_unc > 1:
                total_aleat_unc_pred = np.zeros((n_test, n_features))
            else:
                total_aleat_unc_pred = np.zeros(n_test)

        for batch_idx, batch in enumerate(range(0, n_test, self.batch_size)):
            start_idx = batch
            end_idx = batch + self.batch_size

            x = X_test[start_idx:end_idx]

            exclude = 0
            if len(x) < self.batch_size:
                exclude = self.batch_size - len(x)
                x = np.pad(x, ((0, exclude), (0, 0), (0, 0)), 'constant')

            p, al_un = sess.run([self.out_layer, self.noise_out_layer], feed_dict={
                self.input_pl: x,
                self.dropout_pl: self.test_dropout})

            if self.use_uncs and self.n_neurons_aleat_unc <= 1:
                al_un = al_un.flatten()

            if exclude > 0:
                total_pred[start_idx:end_idx] = p[:-exclude]
                if self.use_uncs:
                    total_aleat_unc_pred[start_idx:end_idx] = al_un[:-exclude]
            else:
                total_pred[start_idx:end_idx] = p
                if self.use_uncs:
                    total_aleat_unc_pred[start_idx:end_idx] = al_un

        if not self.use_uncs:
            total_aleat_unc_pred = None

        return total_pred, total_aleat_unc_pred

    def diff(self, j):
        '''
        @return format [batch_size]
        '''
        t1 = tf.norm(self.pred_mean_pl - self.output_pl)
        t3 = tf.sqrt(2.0)
        t4 = erfinv(j)
        return tf.cast(tf.reduce_mean(t1 - tf.sqrt(self.pred_var_pl) * t3 * t4), tf.float32)

    def acc(self, j):
        '''
        @return format [batch_size]
        '''
        t1 = tf.norm(self.pred_mean_pl - self.output_pl)
        t2 = tf.sqrt(self.pred_var_pl) * tf.sqrt(2.0) * erfinv(j)

        t = t1 < t2
        s = tf.count_nonzero(t)

        b_size = tf.size(t)
        b_size = tf.cast(b_size, tf.int64)

        return tf.cast(s / b_size, tf.float32)

    def predictive_mean(self, preds):
        '''
        Gets input of mc runs for whole batch. Computes predictive mean 
        separately for each input item.

        @param preds: format [n_mc_runs, batch_size, dims]
        @return format [batch_size, dims]
        '''
        m = tf.reduce_mean(preds, axis=0)
        return m

    def predictive_variance(self, pred_mean, preds, al_uncs):
        return tf.reduce_mean(al_uncs + tf.square(preds), axis=0) - tf.square(pred_mean)

    def get_pred_var_and_mean(self, preds, al_uncs, out_layer, noise_out_layer):
        '''
        Computes predictive mean and variance.

        @param preds: format [n_mc_runs, batch_size, dims]
        @param al_uncs: format [n_mc_runs, batch_size, dims]
        @param out_layer: format [batch_size, dims]
        @param noise_out_layer: format [batch_size, dims]
        @return format [batch_size, dims]
        '''
        # append results of last MC run to results from previous runs
        preds = tf.concat([preds, [out_layer]], axis=0)
        al_uncs = tf.concat([al_uncs, [noise_out_layer]], axis=0)

        # compute predictive mean and variance
        pred_mean = self.predictive_mean(preds)
        pred_var = self.predictive_variance(pred_mean, preds, al_uncs)
        return pred_mean, pred_var
