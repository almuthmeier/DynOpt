'''
Same as the class TFRNN, but this class at hand does not allow resetting/saving
the LSTM state in order to allow both: usage of LSTMCell and BasicRNNCell.

(TFRNN can only used with LSTMCell, otherwise exception)
 
Created on Nov 9, 2018

@author: ameier
'''
import math
import sys
import warnings

import numpy as np


class TFRNNWithoutState():
    def __init__(self,
                 n_features,
                 n_time_steps_to_use=7,
                 train_b_size=128,
                 test_b_size=128,
                 n_epochs=5,
                 has_time_outputs=False,
                 custom_reset=False,
                 n_rnn_layers=1,
                 n_neurons=10,
                 rnn_type="LSTM"):
        import tensorflow as tf
        # TODO: n_neurons als Liste mit einem Eintrag pro Schicht

        # parameters for model
        # https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn

        self.rnn_type = rnn_type
        self.n_rnn_layers = n_rnn_layers
        self.act_func = tf.tanh
        self.n_time_steps_to_use = n_time_steps_to_use

        # true if the RNN has for each time slice one output
        self.has_time_outputs = has_time_outputs

        # default value for keep_*-probabilities is for testing/training
        self.in_keep_prob_pl = tf.placeholder_with_default(1.0, ())
        self.out_keep_prob_pl = tf.placeholder_with_default(1.0, ())
        # "Probability that dropping out recurrent connections."
        # https://stackoverflow.com/questions/47415036/tensorflow-how-to-use-variational-recurrent-dropout-correctly
        self.st_keep_prob_pl = tf.placeholder_with_default(1.0, ())

        self.data_type = tf.float32  # train_in.dtype (type of training data)
        self.n_features = n_features  # problem dimensionality
        #self.n_neurons = n_neurons
        self.n_neurons = math.ceil(self.n_features * 1.5)

        # training/testing parameters
        self.train_b_size = train_b_size
        self.test_b_size = test_b_size
        self.n_epochs = n_epochs
        # https://stackoverflow.com/questions/47257518/valueerror-tensor-tensorconst0-shape-dtype-float32-may-not-be-fed-wit
        # (22.9.18)
        self.b_size_pl = tf.placeholder_with_default(self.train_b_size, [])

        # example for placeholder found here:
        # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/dynamic_rnn.py
        # format [batch_size, n_time_steps, n_features]
        # None for b_size -> variable batch size
        # None for n_time_steps -> variable sequence length (for dynamic_rnn)
        self.input_pl = tf.placeholder("float", [None, None, self.n_features])
        if self.has_time_outputs:
            # format [batch_size, n_time_steps, n_features]
            self.output_pl = tf.placeholder(
                "float", [None, None, self.n_features])
        else:
            # format [batch_size, n_features]
            self.output_pl = tf.placeholder("float", [None, self.n_features])

        self.loss = None  # is defined in build_architecture()
        self.train_op = None
        self.out_layer = None
        self.rnn_outputs = None

        self.__build_architecture()

    def __build_architecture(self):
        import tensorflow as tf
        # RNN layers

        # create own cells for the layers (5.10.18)
        # https://stackoverflow.com/questions/48372994/multirnn-and-static-rnn-error-dimensions-must-be-equal-but-are-256-and-129?rq=1
        # https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/
        cell_list = []
        for layer_idx in range(self.n_rnn_layers):
            # TODO evtl CudnnRNNTanh Layer statt BasicRNNCell nutzen
            # https://www.tensorflow.org/api_docs/python/tf/contrib/cudnn_rnn/CudnnRNNTanh
            # https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/BasicRNNCell
            if self.rnn_type == "LSTM":
                rnn_layer = tf.nn.rnn_cell.LSTMCell(
                    num_units=self.n_neurons, activation=self.act_func)
            elif self.rnn_type == "RNN":
                rnn_layer = tf.nn.rnn_cell.BasicRNNCell(
                    num_units=self.n_neurons, activation=self.act_func)
            else:
                warnings.warn("unsupported RNN type: ", self.rnn_type)
            # Dropout for RNN
            # https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/DropoutWrapper
            # input_size: TensorShape object
            # example for input_size found here:
            # https://github.com/tensorflow/tensorflow/issues/11650 (24.9.18)
            if layer_idx == 0:
                input_size = self.n_features  # number of dimensions/features
            else:
                input_size = self.n_neurons
            drop_with_rnn_layer = tf.nn.rnn_cell.DropoutWrapper(cell=rnn_layer,
                                                                input_keep_prob=self.in_keep_prob_pl,
                                                                output_keep_prob=self.out_keep_prob_pl,
                                                                state_keep_prob=self.st_keep_prob_pl,
                                                                variational_recurrent=True,
                                                                dtype=self.data_type,
                                                                input_size=input_size)
            cell_list.append(drop_with_rnn_layer)

        # stack layers together
        cells = tf.nn.rnn_cell.MultiRNNCell(cell_list)

        # dynamic_rnn (oder static_rnn) scheinbar notwendig, weil man ansonsten RNN-Ausgabe nicht in
        # dense-layer schieben kann.

        # https://github.com/tensorflow/tensorflow/issues/8246
        arr = tf.convert_to_tensor(np.array([[self.n_time_steps_to_use]]))
        # https://www.tensorflow.org/api_docs/python/tf/keras/backend/repeat
        s_length = tf.keras.backend.repeat(
            arr, self.b_size_pl)  # like numpy.repeat()
        # convert 2d array to 1d array
        s_length = tf.reshape(s_length, [-1])
        #s_length = [n_time_steps_to_use] * b_size

        # format of outputs: [batch_size, max_time, cell_state_size]
        self.rnn_outputs, _ = tf.nn.dynamic_rnn(
            cell=cells,
            dtype=self.data_type,
            sequence_length=s_length,  # separately for batch
            inputs=self.input_pl)

        # linear output layer (if only next step is predicted, than use
        # (outputs[:, -1, :]) instead)
        # self.out_layer = tf.layers.Dense(
        #    units=self.n_features)(self.rnn_outputs)

        # idea for outputs[:, -1, :] from here:
        # https://stackoverflow.com/questions/42513613/tensorflow-dynamic-rnn-regressor-valueerror-dimension-mismatch
        # (22.9.18) (nur Ausgabe für letzten Zeitschritt nehmen)

        if self.has_time_outputs:
            self.out_layer = tf.keras.layers.TimeDistributed(
                tf.layers.Dense(units=self.n_features))(self.rnn_outputs)
        else:
            self.out_layer = tf.layers.Dense(
                units=self.n_features)(self.rnn_outputs[:, -1, :])

        # loss definition
        self.loss = tf.losses.mean_squared_error(
            labels=self.output_pl, predictions=self.out_layer)
        # optimizer definition
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=0.001).minimize(self.loss)

        # in order to resume training later
        # https://www.tensorflow.org/api_guides/python/meta_graph
        #tf.add_to_collection('train_op', self.train_op)

    def train(self, sess, train_in, train_out, in_keep_prob=1.0, out_keep_prob=1.0, st_keep_prob=1.0,
              shuffle_between_epochs=False, saver=None, saver_path=None, model_name=None,
              do_validation=False, do_early_stopping=False, validation_in=None, validation_out=None):

        import tensorflow as tf
        n_train_data = train_out.shape[0]
        # TODO hier oder im Aufrufer tun?
        sess.run(tf.global_variables_initializer())

        train_err_per_epoch = []
        val_err_per_epoch = []
        min_val_err = sys.float_info.max  # maximum possible float value
        # train in epochs
        for i in range(self.n_epochs):
            print("\nepoch: ", i, flush=True)

            # shuffle training data if desired
            if shuffle_between_epochs:
                # generate and shuffle indices to shuffle input and output data
                # in same order
                idx = np.arange(n_train_data)
                np.random.shuffle(idx)
                train_in = train_in[idx]
                train_out = train_out[idx]
            # https://stackoverflow.com/questions/46840539/online-or-batch-training-by-default-in-tensorflow
            # (21.9.18)
            # train in batches
            # also last batch with possibly fewer data is applied
            for start in range(0, n_train_data, self.train_b_size):
                end = min(start + self.train_b_size, n_train_data + 1)
                in_data = train_in[start:end]
                out_data = train_out[start:end]
                # b_size may be different for batches
                curr_b_size = len(in_data)

                _, loss_value, o_val, rnn_out = sess.run([self.train_op, self.loss, self.out_layer,
                                                          self.rnn_outputs],
                                                         feed_dict={
                    self.input_pl: in_data,
                    self.output_pl: out_data,
                    self.b_size_pl: curr_b_size,
                    self.in_keep_prob_pl: in_keep_prob,
                    self.out_keep_prob_pl: out_keep_prob,
                    self.st_keep_prob_pl: st_keep_prob})  # keep int. state
                #print("in: ", in_data)
                #print("real_out: ", out_data)
                #print("output: ", o_val)
                #print("Loss: ", loss_value)
                # cell_state_size is equal number neurons
                # print(
                #    "rnn_out_shape [batch_size, time_steps, cell_state_size]: ", rnn_out.shape)
            # save model
            if saver is not None:
                name = saver_path + model_name + str(i) + ".ckpt"
                save_path = saver.save(sess, name)
                #print("Model saved in path: %s" % save_path)
                # for i, var in enumerate(saver._var_list):
                #    print('Var {}: {}'.format(i, var))

            # training error
            train_error, train_model_out = self.evaluate_model(
                train_in, sess, self.train_b_size, True, train_out)
            train_err_per_epoch.append(train_error)
            # validation error
            if do_validation and validation_in is not None and validation_out is not None:
                validation_error, _ = self.evaluate_model(
                    validation_in, sess, self.train_b_size, True, validation_out)
                val_err_per_epoch.append(validation_error)
                min_val_err = min(min_val_err, validation_error)
                if do_early_stopping:
                    # compute generalization loss
                    gloss = (validation_error / min_val_err - 1)
                    print("train(): min_val", min_val_err)
                    print("train(): gloss: ", gloss)
                    if gloss > 0.1:  # TODO(dev)
                        # stop training
                        print(
                            "early stopping: stopped training at epoch: ", i, flush=True)
                        break
            else:
                validation_error = None
                val_err_per_epoch = None
            #print("train(): train_error: ", train_error, flush=True)
            #print("train(): validation_error: ", validation_error, flush=True)
        # TODO which state should be returned (after training/evaluation)?
        # TODO iis train_error a list/array????
        return np.array(train_error), np.array(train_model_out), validation_error, train_err_per_epoch, val_err_per_epoch

    def test(self, sess, test_in, test_out, in_keep_prob=1.0, out_keep_prob=1.0, st_keep_prob=1.0):
        test_error, test_model_out = self.evaluate_model(test_in,
                                                         sess, self.test_b_size, True, test_out, in_keep_prob, out_keep_prob, st_keep_prob)
        print("test(): test_error: ", test_error, flush=True)
        return test_error, test_model_out

    def evaluate_model(self, input_data, sess, b_size, compute_loss=False, output_data=None,
                       in_keep_prob=1.0, out_keep_prob=1.0, st_keep_prob=1.0):
        '''
        If compute_loss True then output_data must not be None 
        '''
        n_data = len(input_data)
        batch_errors = []
        model_outputs = []

        # prepare session goals
        if not compute_loss:
            session_goals = [self.out_layer]
        else:
            session_goals = [self.loss, self.out_layer]

        for start in range(0, n_data, b_size):
            end = min(start + b_size, n_data + 1)
            in_data = input_data[start:end]
            curr_b_size = len(in_data)  # b_size may be different for batches

            # prepare dictionary
            session_dict = {self.input_pl: in_data,
                            self.b_size_pl: curr_b_size,
                            self.in_keep_prob_pl: in_keep_prob,
                            self.out_keep_prob_pl: out_keep_prob,
                            self.st_keep_prob_pl: st_keep_prob}  # keep int. state}

            if compute_loss:
                session_dict[self.output_pl] = output_data[start:end]

            # run session
            session_output = sess.run(session_goals, feed_dict=session_dict)

            # prepare results
            if not compute_loss:
                model_outputs.append(session_output[0])  # only one value
            else:  # tuple of values
                batch_errors.append(session_output[0])
                model_outputs.append(session_output[1])
        flattened_output = [
            pred for batch_result in model_outputs for pred in batch_result]
        if not compute_loss:
            return None, np.array(flattened_output)
        else:
            return np.average(batch_errors), np.array(flattened_output)

    def predict_one_sample(self, sess, in_sample, in_keep_prob=1.0,
                           out_keep_prob=1.0, st_keep_prob=1.0):
        '''
        Predict next step for one single sample.
        '''
        b_size = 1
        model_out = sess.run([self.out_layer], feed_dict={self.input_pl: in_sample,
                                                          self.b_size_pl: b_size,
                                                          self.in_keep_prob_pl: in_keep_prob,
                                                          self.out_keep_prob_pl: out_keep_prob,
                                                          self.st_keep_prob_pl: st_keep_prob})
        # return value of sees.run() is always a list, even if there is only one goal
        # -> use model_out[0]
        return np.array(model_out[0])

    def predict_and_train_stepwise(self, sess, train_in, train_out, in_data, out_data,
                                   pred_uncer=False, obs_noise=None, n_mc_runs=None,
                                   in_keep_prob=1.0, out_keep_prob=1.0, st_keep_prob=1.0):
        '''
        Predicts each input data separately and is trained anew after each
        prediction.
        @param train_in: input data on that the model was trained
        @param train_out: output data on that the model was trained

        # TODO man könnte variabel lassen, auf wie vielen Daten neu trainiert werden soll. 
        '''
        n_data = in_data.shape[0]
        predictions = []
        pred_means = []
        pred_vars = []
        n_sampl = 1  # should be 1
        b_size = 1

        counter = 0
        for (sample_in, sample_out) in zip(in_data, out_data):
            print("sample ", counter, " of ", n_data, flush=True)
            counter += 1
            # prediction of next step

            resh_sample_x = sample_in.reshape(
                n_sampl, self.n_time_steps_to_use, self.n_features)
            sample_y_hat = self.predict_one_sample(
                sess, resh_sample_x, in_keep_prob, out_keep_prob, st_keep_prob)
            # sample_y_hat is 3-dimensional list
            predictions.append(sample_y_hat[0])

            # TODO: Implementierung mit Tensoren/arrays effizient?
            # training with real value of next step to prepare for next prediction
            # add data to old data, skip first/oldest one
            train_in = np.delete(train_in, 0, axis=0)
            train_in = np.append(train_in, resh_sample_x, axis=0)
            train_out = np.delete(train_out, 0, axis=0)
            train_out = np.append(
                train_out, np.array([sample_out]), axis=0)

            # TODO braucht train hier evtl. den state???
            train_error, train_model_out = self.train(
                sess, in_keep_prob, out_keep_prob, st_keep_prob)

            # predictive uncertainty
            if pred_uncer:
                assert n_mc_runs is not None
                assert obs_noise is not None
                mc_model_outputs, pred_mean, pred_var = self.predict_uncertainty(sess, n_mc_runs, obs_noise,
                                                                                 resh_sample_x, in_keep_prob,
                                                                                 out_keep_prob, st_keep_prob)  # , np.reshape(sample_out, (n_sampl, -1)))
                pred_means.append(pred_mean)
                pred_vars.append(pred_var)
        # predictions is list of 2-dimensional arrays (therefore has to be
        # converted from 3d to 2d)
        flattened_predictions = [pred for arr in predictions for pred in arr]
        flattened_pred_means = [m for arr in pred_means for m in arr]
        flattened_pred_vars = [v for arr in pred_vars for v in arr]
        return np.array(flattened_predictions), mc_model_outputs, np.array(flattened_pred_means), np.array(flattened_pred_vars)

    def predict_multiple_steps(self, n_steps, in_sample, sess, in_keep_prob=1.0, out_keep_prob=1.0, st_keep_prob=1.0):
        '''
        Predicts next step for sample and uses it for predicting the next step.
        This way n_steps are predicted for the sample sequence.
        @param in_sample 3d array, shape [1, n_time_steps_to_use, n_features]
        '''
        predictions = []
        n_sampl = 1
        for _ in range(0, n_steps):
            # reshape input from 2d to 3d
            in_sample = in_sample.reshape(
                n_sampl, self.n_time_steps_to_use, self.n_features)
            # predict next step, output is 3-dimensional
            next_step = self.predict_one_sample(
                sess, in_sample, in_keep_prob, out_keep_prob, st_keep_prob)

            # prepare input for next prediction: delete oldest time step and
            # append new one (i.e. last time step in the predicted sequence)
            in_sample = np.delete(in_sample, 0, axis=1)

            if self.has_time_outputs:
                next_step = np.array(next_step)
                pred = next_step[:, -1, :]
                # make 3d array from 2d array (otherwise .append wouldn't work)
                pred = np.array([pred])
                in_sample = np.append(in_sample, pred, axis=1)
            else:
                pred = np.array([next_step])
                in_sample = np.append(in_sample, pred, axis=1)
            # alternative: use prediction for all time steps as next input
            #in_sample = next_step

            # store prediction
            predictions.append(next_step)

        # reshape predictions from 4d list to 2d list (list of predictions),
        # takes last step in predicted sequences as output
        flattended_preds = [batch_array[0][-1] for batch_array in predictions]
        return np.array(flattended_preds)

    def predict_uncertainty(self, sess, n_mc_runs, obs_noise, in_data,
                            in_keep_prob=1.0, out_keep_prob=1.0, st_keep_prob=1.0):

        b_size = self.test_b_size
        # man bekommt also für jede Dimension/Feature eine
        # Unsicherheitsschätzung?!
        model_outputs = []
        for _ in range(n_mc_runs):
            _, model_out = self.evaluate_model(
                in_data, sess, b_size, False, None, in_keep_prob, out_keep_prob, st_keep_prob)
            model_outputs.append(model_out)

        # predictive mean
        pred_mean = np.average(model_outputs, axis=0)
        # predictive variance
        summed_values = 0
        for i in range(n_mc_runs):
            summed_values += obs_noise
            summed_values += np.square(model_outputs[i])
        pred_var = summed_values / n_mc_runs - np.square(pred_mean)
        return np.array(model_outputs), np.array(pred_mean), np.array(pred_var)
# TODO:
# + Testen
# + wie berechnet man Testfehler? Mittelwert über Batches
# + wie berechnet man Trainingsfehler?
# + Trainingsfehler ausgeben
# + warum ist regression sooo schlecht?
# - TODOs in anderer Datei
# + Visualisierung der Vorhersage
# + Was, wenn Batchsize kein Teiler der Trainings-/Testdaten ist? Weglassen oder kleinerer letzter Batch?
# + feiner aufgelöste Zeitreihe fürs Testen
# + bei Sinuskurve (mit feinerer Auflösung) überschneiden sich Trainings- u. Testdaten?!
# + useclass: skalieren,
