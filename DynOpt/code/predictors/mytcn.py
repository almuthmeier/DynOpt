'''
Builds a complete Temporal Convolution Network (with output layer, train and 
test functions). 

Main parts of code taken from add_test.py from: 
https://github.com/YuanTingHsieh/TF_TCN/blob/master/adding_problem/add_test.py

Modified code to enable Monte Carlo dropout: 
    - placeholder for dropout (now dropout is also possible during testing)
'''
from pathlib import Path
import time

from sklearn.metrics.pairwise import paired_distances

import numpy as np
from predictors.tcn import TemporalConvNet
import tensorflow as tf


print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())


class MyTCN():

    def __init__(self, in_channels, output_size, num_channels, sequence_length,
                 kernel_size, default_dropout, batch_size, init=False, use_aleat_unc=False):
        '''
        @param default_dropout: is overwritten in train() and evaluate()
        @param aleat_unc: True if the loss function for aleatoric uncertainty 
        should be applied
        '''
        self.lr = 4e-3  # TODO
        clip = -1
        self.batch_size = batch_size

        self.use_aleat_unc = use_aleat_unc
        n_classes = output_size
        # None instead of batch_size
        self.input_pl = tf.placeholder(
            tf.float32, (None, sequence_length, in_channels))
        # None instead of batch_size
        self.output_pl = tf.placeholder(
            tf.float32, (None, n_classes))
        # default value for dropout (is for testing/training)
        self.dropout_pl = tf.placeholder_with_default(default_dropout, ())

        tcn = TemporalConvNet(input_layer=self.input_pl, num_channels=num_channels, sequence_length=sequence_length,
                              kernel_size=kernel_size, dropout=self.dropout_pl, init=init)

        # output layer for prediction
        self.out_layer = tf.contrib.layers.fully_connected(tcn[:, -1, :], output_size, activation_fn=None,
                                                           weights_initializer=tf.initializers.random_normal(0, 0.01))

        # deviation of prediction and labels
        prediction_mse = tf.losses.mean_squared_error(
            labels=self.output_pl, predictions=self.out_layer)

        # output layer for observation noise
        if self.use_aleat_unc:
            # one neuron because one uncertainty output for one input
            self.n_neurons_aleat_unc = 1
            # for each dimension one uncertainty output
            self.n_neurons_aleat_unc = output_size
            # heteroscedastic aleatoric uncertainty
            self.noise_out_layer = tf.contrib.layers.fully_connected(tcn[:, -1, :], self.n_neurons_aleat_unc, activation_fn=tf.nn.softplus,
                                                                     weights_initializer=tf.initializers.random_normal(0, 0.01))

            # compute loss
            t1 = tf.scalar_mul(1 / 2, tf.exp(-self.noise_out_layer))
            t2 = tf.multiply(t1, prediction_mse)
            t3 = tf.scalar_mul(1 / 2, self.noise_out_layer)
            t4 = tf.add(t2, t3)
            self.loss = tf.reduce_mean(t4)
        else:
            self.noise_out_layer = []
            self.loss = prediction_mse

        # output
        tf.summary.scalar('mse', self.loss)
        self.merged = tf.summary.merge_all()

        # optimization operation
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        if clip > 0:
            gradients, _ = tf.clip_by_global_norm(gradients, clip)
        self.update_step = optimizer.apply_gradients(zip(gradients, variables))

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

    def train(self, n_epochs, sess, X_train, Y_train, n_train, log_interval, train_writer, dropout):
        for ep in range(1, n_epochs + 1):
            steps = 0
            total_loss = 0
            start_time = time.time()

            for batch_idx, indices in self.index_generator(n_train):
                x = X_train[indices]
                y = Y_train[indices]

                summary, _, p, al_un, l = sess.run([self.merged, self.update_step,
                                                    self.out_layer, self.noise_out_layer, self.loss],
                                                   feed_dict={self.input_pl: x, self.output_pl: y,
                                                              self.dropout_pl: dropout})
                total_loss += l
                steps += 1
                train_writer.add_summary(summary, steps)

                if (batch_idx > 0 and batch_idx % log_interval == 0):
                    avg_loss = total_loss / log_interval
                    elapsed = time.time() - start_time
                    print('| Epoch {:3d} | {:5d}/{:5d} batches | lr {:2.5f} | ms/batch {:5.2f} | '
                          'loss {:5.8f} |'.format(
                              ep, batch_idx, n_train // self.batch_size +
                              1, self.lr, elapsed * 1000 / log_interval,
                              avg_loss), flush=True)
                    start_time = time.time()
                    total_loss = 0

    def evaluate(self, sess, X_test, Y_test, n_test, dropout):

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

            p, al_un, l = sess.run([self.out_layer, self.noise_out_layer, self.loss], feed_dict={
                self.input_pl: x, self.output_pl: y,
                self.dropout_pl: dropout})

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

        if not self.use_aleat_unc:
            total_aleat_unc_pred = None

        distances = paired_distances(Y_test, total_pred)
        return total_pred, total_aleat_unc_pred, distances

    def predict(self, sess, X_test, n_test, n_features, dropout):
        '''
        Only prediction, no computation of MSE.
        @param X_test: 3d array, format [n_data, n_time_steps, n_features]
        '''
        total_pred = np.zeros((n_test, n_features))
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
                self.dropout_pl: dropout})

            if self.n_neurons_aleat_unc <= 1:
                al_un = al_un.flatten()

            if exclude > 0:
                total_pred[start_idx:end_idx] = p[:-exclude]
                total_aleat_unc_pred[start_idx:end_idx] = al_un[:-exclude]
            else:
                total_pred[start_idx:end_idx] = p
                total_aleat_unc_pred[start_idx:end_idx] = al_un

        if not self.use_aleat_unc:
            total_aleat_unc_pred = None

        return total_pred, total_aleat_unc_pred
