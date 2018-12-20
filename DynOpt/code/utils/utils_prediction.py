'''
Creates:
    - the prediction models, including our proposed RNN predictor
    - the training data for the prediction models
    - the prediction 
    
Created on Jan 18, 2018

@author: ameier
'''

import math
import warnings

from sklearn.preprocessing.data import MinMaxScaler

import numpy as np


def make_multidim_samples_from_series(train_data, n_time_steps):
    '''
    Converts the training data into the right format as supervised learning
    problem. 
    @param train_data: 2d numpy array: one row for each time step
    @param n_time_steps: number of time steps to use for prediction/training
    @return: 3d numpy array with shape #samples,#n_time_steps+1,#n_features:
     one row for each sample: each row consists of a time series with 
     n_time_steps entries:
         [
         [[..,..,..], [...], ...],     # time series 1
         [[..,..,..], [...], ...]      # time series 2
         ]
    '''
    # without last entries
    series = train_data[:-n_time_steps]
    n_samples = series.shape[0]
    n_features = series.shape[1]
    # reshape series to 3-dimensional array: each of the n_samples rows
    # consists of one entry with n_features (the first observation of the time
    # series)
    series = series.reshape(n_samples, 1, n_features)

    # every time skip one entry more from the beginning and the end
    for i in range(1, n_time_steps):
        tmp = train_data[i:-(n_time_steps - i)]
        tmp = tmp.reshape(n_samples, 1, n_features)
        series = np.concatenate((series, tmp), axis=1)

    # without first entries
    last = train_data[n_time_steps:]
    last = last.reshape(n_samples, 1, n_features)
    series = np.concatenate((series, last), axis=1)
    return series


def shuffle_split_output(samples, returnseq, ntimesteps, n_features, shuffle):
    '''
    Shuffle data (the rows, but content within rows stays same). Cuts also the
    data that doesn't fit into the last batch.
    '''
    if shuffle:
        np.random.shuffle(samples)

    # split input and output
    in_data = samples[:, :-1]
    if returnseq:
        # return values for all time steps
        out_data = samples[:, 1:]
    else:
        out_data = samples[:, -1]
    n_train_data = len(in_data)

    # convert input format to 3d array [n_train_samples, time_steps, features]
    in_data = in_data.reshape([n_train_data, ntimesteps, n_features])

    # convert output format, depending on whether a sequence is returned
    if returnseq:
        # 3d array
        out_data = out_data.reshape([n_train_data, ntimesteps, n_features])
    else:
        # 2d arr
        out_data = out_data.reshape([n_train_data, n_features])
    return np.float32(in_data), np.float32(out_data)


def build_predictor(mode, n_time_steps, n_features, batch_size, n_neurons,
                    returnseq, apply_tl, n_overall_layers, epochs, rnn_type,
                    ntllayers, with_dense_first, tl_learn_rate):
    '''
    Creates the desired prediction model.
    @param mode: which predictor: no, rnn, autoregressive, tfrnn, tftlrnn,tftlrnndense
    @param batch_size: batch size for the RNN
    @param n_time_steps: number of time steps to use for prediction/training
    @param n_features: dimensionality of the solution space
    '''
    if mode == "no":
        predictor = None
    elif mode == "rnn":
        # our proposed RNN as predictor (EvoApps 2018 paper)
        from keras.layers.core import Dense
        from keras.layers.recurrent import SimpleRNN
        from keras.models import Sequential
        predictor = Sequential()
        predictor.add(SimpleRNN(n_neurons, activation='tanh', batch_input_shape=(
            batch_size, n_time_steps, n_features), stateful=True))
        predictor.add(Dense(units=n_features))
        predictor.compile(loss='mean_squared_error', optimizer='adam')
    elif mode == "autoregressive":
        # no object required since the AR model is created every time a
        # prediction is needed
        predictor = None
    elif mode == "tfrnn" or mode == "tftlrnn" or mode == "tftlrnndense":
        from utils.utils_transferlearning import build_tl_rnn_predictor
        predictor = build_tl_rnn_predictor(rnn_type, ntllayers,
                                           n_overall_layers, n_time_steps, epochs, n_features,
                                           returnseq, batch_size, apply_tl, with_dense_first,
                                           tl_learn_rate)
    else:
        msg = "unknown prediction mode " + mode
        warnings.warn(msg)
    return predictor


def predict_with_autoregressive(new_train_data, n_features, n_time_steps, scaler):
    '''
    Predicts next optimum position with autoregressive model.
    @param new_train_data: 

    Throws ValueError (if training data consists data which are nearly zero) 

    http://www.statsmodels.org/devel/tsa.html#univariate-autogressive-processes-ar (overview)
    http://www.statsmodels.org/devel/generated/statsmodels.tsa.ar_model.AR.html#statsmodels.tsa.ar_model.AR (AR)
    http://www.statsmodels.org/devel/vector_ar.html#var (VAR)
    http://www.statsmodels.org/devel/generated/statsmodels.tsa.vector_ar.var_model.VAR.html#statsmodels.tsa.vector_ar.var_model.VAR (VAR)

    https://qiita.com/vinyip918/items/583ac76b9b05cdcc2dde (example for AR)
    '''
    # number of time steps to predict (1 because only next optimum)
    n_prediction_steps = 1
    # maximum number of time steps to use for prediction
    # -1 for VAR, otherwise exception (perhaps because otherwise each column has only one value
    # https://datascience.stackexchange.com/questions/38203/var-model-valueerror-x-already-contains-a-constant
    max_lag = n_time_steps - 1
    n_train_data = len(new_train_data)
    if n_features == 1:
        from statsmodels.tsa.ar_model import AR
        # train autoregression
        new_train_data = new_train_data.flatten()
        model = AR(new_train_data)
        # throws exception:  "ValueError: x already contains a constant" if the
        # training data contain values similar to zero
        try:
            model_fit = model.fit(maxlag=max_lag)
        except ValueError:
            # "ValueError: shapes (4,) and (5,) not aligned: 4 (dim 0) != 5 (dim 0)"
            # if max_lag = n_time_steps
            raise
        # https://stackoverflow.com/questions/31665256/python-statsmodels-params-parameter-for-predict-function-of-arima-models
        predictions = model_fit.predict(start=n_train_data,
                                        end=n_train_data + n_prediction_steps - 1)
    else:
        # for multivariate data, the VAR type does not work for univariate data
        from statsmodels.tsa.vector_ar.var_model import VAR

        # train autoregression
        model = VAR(new_train_data)
        # throws exception:  "ValueError: x already contains a constant" if the
        # training data contain values similar to zero
        try:
            model_fit = model.fit(maxlags=max_lag)
        except ValueError:
            # "ValueError: x already contains a constant"
            # if max_lag = n_time_steps
            model_fit = model.fit()
            print("ARR prediction: caught Exception", flush=True)

        lag_order = model_fit.k_ar
        #print('Lag: %s' % lag_order)
        #print('Coefficients: %s' % model_fit.params)

        # make prediction, idea for "forecast" instead of "predict" from here:
        # http://www.statsmodels.org/0.6.1/vector_ar.html (7.10.17)
        # http://www.statsmodels.org/dev/vector_ar.html
        predictions = model_fit.forecast(
            new_train_data[-lag_order:], n_prediction_steps)

    if n_prediction_steps == 1:
        # make 1d numpy array from 2d array if only one prediction is done
        prediction = predictions.flatten()

    # invert scaling (1d array would result in DeprecatedWarning -> pass 2d)
    converted = scaler.inverse_transform(np.array([prediction]))
    return converted.flatten()


def predict_with_rnn(new_train_data, noisy_series, n_epochs, batch_size, n_time_steps,
                     n_features, scaler, predictor):
    '''
    Predicts next optimum position with a recurrent neural network.
    '''
    #========================
    # prepare training data

    # make supervised data from series
    train_samples = make_multidim_samples_from_series(
        new_train_data, n_time_steps)
    if noisy_series is not None:
        pass  # TODO implement

    # separate input series (first values) and prediction value (last value)
    train_in_data, train_out_data = train_samples[:, :-1], train_samples[:, -1]

    # define constants from data
    n_train_samples = train_in_data.shape[0]
    #n_features = train_out_data.shape[1]

    # Samples/TimeSteps/Features format, for example:
    train_in_data = train_in_data.reshape(
        n_train_samples, n_time_steps, n_features)

    #========================
    # train regressor

    for i in range(n_epochs):
        hist = predictor.fit(train_in_data, train_out_data, epochs=1,
                             batch_size=batch_size,  verbose=0, shuffle=False)  # TODO shuffle should be True
        # print("epoch ", i, "/", n_epochs, ": loss ",
        #      hist.history['loss'], flush=True)
        predictor.reset_states()
        # states usually are reset after epochs:
        # https://stackoverflow.com/questions/45623480/stateful-lstm-when-to-reset-states
        # https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

    # predict on training set to set state for prediction on test set
    #train_prediction = predictor.predict(train_in_data, batch_size=batch_size)

    #========================
    # prediction for next step (with n_time_steps)
    prediction_series = np.array(new_train_data[-n_time_steps:])

    n_sampl = 1  # should be 1
    reshaped_sample_x = prediction_series.reshape(
        n_sampl, n_time_steps, n_features)
    sample_y_hat = predictor.predict(
        reshaped_sample_x, batch_size=batch_size)

    # invert scaling
    next_optimum = scaler.inverse_transform(sample_y_hat).flatten()
    return next_optimum


def predict_with_tfrnn(sess, new_train_data, noisy_series, n_epochs, batch_size, n_time_steps,
                       n_features, scaler, predictor, returnseq, shuffle):
    '''
    Predicts next optimum position with a tensorflow recurrent neural network.
    @param new_train_data: [n_chgperiods, dims] (if differences are predicted 
    the format is: [n_chgperiods-1, dims]
    @param noisy_series: [n_series, n_chgperiods, dims]
    '''
    #========================
    # prepare training data

    # make supervised data from series [#samples,#n_time_steps+1,#n_features]
    train_samples = make_multidim_samples_from_series(
        new_train_data, n_time_steps)
    if noisy_series is not None:
        # 4d array [#series, #samples, #n_time_steps+1, #n_features]
        noisy_samples = np.array([make_multidim_samples_from_series(
            noisy_series[i], n_time_steps) for i in range(len(noisy_series))])
        # convert 4d to 3d array [#series*#samples,#n_time_steps+1,#n_features]
        noisy_samples = np.reshape(
            noisy_samples, (-1, n_time_steps + 1, n_features))
        # append noisy samples to "real" samples -> 3d array
        train_samples = np.concatenate((train_samples, noisy_samples))

    # separate input series (first values) and prediction value (last value)
    train_in_data, train_out_data = shuffle_split_output(train_samples, returnseq,
                                                         n_time_steps, n_features, shuffle)
    #========================
    # train regressor
    # TODO save model? report training error?
    keep_prob = 0.95
    train_error, _, _, train_err_per_epoch, _ = predictor.train(sess, train_in_data, train_out_data, in_keep_prob=keep_prob, out_keep_prob=keep_prob, st_keep_prob=keep_prob,
                                                                shuffle_between_epochs=True, saver=None, saver_path=None, model_name=None,
                                                                do_validation=False, do_early_stopping=False, validation_in=None, validation_out=None)

    #========================
    # prediction for next step (with n_time_steps)
    prediction_series = np.array(new_train_data[-n_time_steps:])

    n_sampl = 1  # should be 1
    reshaped_sample_x = prediction_series.reshape(
        n_sampl, n_time_steps, n_features)

    sample_y_hat = predictor.predict_one_sample(sess, reshaped_sample_x, in_keep_prob=1.0,
                                                out_keep_prob=1.0, st_keep_prob=1.0)
    if returnseq:
        # cut previous time steps that are also returned in this case
        sample_y_hat = sample_y_hat[:, -1, :]

    # invert scaling
    next_optimum = scaler.inverse_transform(sample_y_hat).flatten()
    return next_optimum, train_error, train_err_per_epoch


def predict_next_optimum_position(mode, sess, new_train_data, noisy_series, n_epochs, batch_size,
                                  n_time_steps, n_features, scaler, predictor,
                                  returnseq, shuffle):
    '''
    @param mode: the desired predictor
    @param new_train_data: 2d numpy array: contains time series of 
    (n_time_steps+1) previous found solutions
    @param n_epochs: number training epochs for RNN 
    @param batch_size: batch size for RNN
    @param n_time_steps: number of previous solutions to use for the prediction
    @param n_features: dimensionality of the solution space
    @param scaler: scaler to re-scale the prediction (because the training data 
    is already scaled)
    @param predictor: RNN predictor object to use for the prediction, so that
    it is not required to train it completely new
    @return 1d numpy array: position of next optimum (already re-scaled)
    '''
    train_error = None
    train_err_per_epoch = None
    if mode == "rnn":
        prediction = predict_with_rnn(new_train_data, noisy_series, n_epochs, batch_size,
                                      n_time_steps, n_features, scaler, predictor)
    elif mode == "autoregressive":
        try:
            prediction = predict_with_autoregressive(
                new_train_data, n_features, n_time_steps, scaler)
        except ValueError:
            raise
    elif mode == "no":
        prediction = None
    elif mode == "tfrnn" or mode == "tftlrnn" or mode == "tftlrnndense":
        prediction, train_error, train_err_per_epoch = predict_with_tfrnn(sess, new_train_data, noisy_series, n_epochs, batch_size, n_time_steps,
                                                                          n_features, scaler, predictor, returnseq, shuffle)
    return prediction, train_error, train_err_per_epoch


def get_n_neurons(n_neurons_type, dim):
    '''
    (number of neurons can not directly be specified as input because it is 
    computed in some cases depending on the problem dimensionality)
    '''
    if n_neurons_type == "fixed20":
        n_neurons = 20
    elif n_neurons_type == "dyn1.3":
        n_neurons = math.ceil(dim * 1.3)
    else:
        msg = "unkwnown type for neuronstype (number of neurons): " + \
            str(n_neurons_type)
        warnings.warn(msg)
    return n_neurons


def prepare_scaler(min_input_value, max_input_value, dim):
    '''
    Instantiates scaler that scales input data into range [-1,1] whereby the
    minimum and maximum input values have to be specified.

    To scale data only the transform() method has to be called (otherwise 
    different scaling behavior would be obtained for different data).
    '''
    # fit scaler to desired range [-1,1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # first row contains minimum value per feature, second row max values
    min_max_per_feature = np.array(
        [[min_input_value] * dim, [max_input_value] * dim])
    scaler.fit(min_max_per_feature)
    return scaler


def get_noisy_time_series(original_series, n_series, stddev_per_chgperiod):
    '''
    Generates n_series variations of original_series disturbed with gaussian 
    noise of strength stddev_per_chgperiod.

    Requires repeated execution of EA for the change periods.

    @param original_series: for each change period the best found solution, 
    format [n_chgperiods, dims]
    @param n_series: number of noisy series to generate
    @param stddev_per_chgperiod: for each change period the standard deviation of
    the best found solution among the runs, format [n_chgperiod, dims]
    @return: nump array, format [n_series, n_chgperiods, dims]
    '''
    n_chgperiods, dims = original_series.shape
    return np.random.normal(loc=original_series, scale=stddev_per_chgperiod,
                            size=(n_series, n_chgperiods, dims))
