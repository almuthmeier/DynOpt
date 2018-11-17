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


def build_predictor(mode, n_time_steps, n_features, batch_size, n_neurons):
    '''
    Creates the desired prediction model.
    @param mode: which predictor: no, rnn, autoregressive, tltfrnn, tfrnn
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
    elif mode == "tfrnn":
        from utils.utils_transferlearning import build_tl_rnn_predictor
        rnn_type = "RNN"
        ntllayers = 0
        n_overall_layers = 2
        epochs = 5
        returnseq = True
        test_b_size = 128
        apply_tl = True
        predictor = build_tl_rnn_predictor(rnn_type, ntllayers,
                                           n_overall_layers, n_time_steps, epochs, n_features,
                                           returnseq, test_b_size, apply_tl)
    elif mode == "tltfrnn":
        from utils.utils_transferlearning import build_tl_rnn_predictor
        rnn_type = "RNN"
        ntllayers = 1
        n_overall_layers = 2
        epochs = 5
        returnseq = True
        test_b_size = 128
        apply_tl = True
        predictor = build_tl_rnn_predictor(rnn_type, ntllayers,
                                           n_overall_layers, n_time_steps, epochs, n_features,
                                           returnseq, test_b_size, apply_tl)
    else:
        msg = "unknown prediction mode " + mode
        warnings.warn(msg)
    return predictor


def predict_with_autoregressive(new_train_data, scaler):
    '''
    Predicts next optimum position with autoregressive model.
    @param new_train_data: 

    Throws ValueError (if training data consists data which are nearly zero) 
    '''
    from statsmodels.tsa.vector_ar.var_model import VAR

    # number of time steps to predict (1 because only next optimum)
    n_prediction_steps = 1
    # train autoregression
    model = VAR(new_train_data)
    # throws exception:  "ValueError: x already contains a constant" if the
    # training data contain values similar to zero
    try:
        model_fit = model.fit()
    except ValueError:
        raise

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


def predict_with_rnn(new_train_data, n_epochs, batch_size, n_time_steps,
                     n_features, scaler, predictor):
    '''
    Predicts next optimum position with a recurrent neural network.
    '''
    #========================
    # prepare training data

    # make supervised data from series
    train_samples = make_multidim_samples_from_series(
        new_train_data, n_time_steps)

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
                             batch_size=batch_size,  verbose=0, shuffle=False)
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


def predict_next_optimum_position(mode, new_train_data, n_epochs, batch_size,
                                  n_time_steps, n_features, scaler, predictor):
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

    if mode == "rnn":
        return predict_with_rnn(new_train_data, n_epochs, batch_size,
                                n_time_steps, n_features, scaler, predictor)
    elif mode == "autoregressive":
        try:
            return predict_with_autoregressive(new_train_data, scaler)
        except ValueError:
            raise
    elif mode == "no":
        return None
    elif mode == "tfrnn":
        pass  # TODO (Almuth)
    elif mode == "tltfrnn":
        pass  # TODO (Almuth)


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
