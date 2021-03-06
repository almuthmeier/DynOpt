'''
Creates:
    - the prediction models, including our proposed RNN predictor
    - the training data for the prediction models
    - the prediction 
    
Created on Jan 18, 2018

@author: ameier
'''
from _collections import OrderedDict
import copy
import math
import warnings

import numpy as np
from predictors.myautomatictcn import MyAutoTCN
from utils import utils_dynopt
from utils.my_scaler import MyMinMaxScaler


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


def shuffle_split_output(samples, returnseq, ntimesteps, n_features, shuffle,
                         pred_np_rnd_generator):
    '''
    Shuffle data (the rows, but content within rows stays same). Cuts also the
    data that doesn't fit into the last batch.
    '''
    if shuffle:
        pred_np_rnd_generator.shuffle(samples)

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


def build_all_predictors(mode, n_time_steps, n_features, batch_size, n_neurons,
                         returnseq, apply_tl, n_overall_layers, epochs, rnn_type,
                         ntllayers, with_dense_first, tl_learn_rate, use_uncs,
                         train_mc_runs, train_dropout, test_dropout, kernel_size,
                         nhid, lr):
    '''
    @return: dictionary containing for each single predictor the respective
    model
    '''
    if "hybrid-" in mode:
        # e.g. "hybrid-autoregressive-rnn"
        parts = mode.split("-")
        predictor_names = parts[1:]  # first entry is "hybrid"
    else:
        predictor_names = [mode]

    predictors = OrderedDict()
    for name in predictor_names:
        predictors[name] = build_predictor(name, n_time_steps, n_features, batch_size, n_neurons,
                                           returnseq, apply_tl, n_overall_layers, epochs, rnn_type,
                                           ntllayers, with_dense_first, tl_learn_rate, use_uncs,
                                           train_mc_runs, train_dropout, test_dropout, kernel_size,
                                           nhid, lr)
    return predictors


def build_predictor(mode, n_time_steps, n_features, batch_size, n_neurons,
                    returnseq, apply_tl, n_overall_layers, epochs, rnn_type,
                    ntllayers, with_dense_first, tl_learn_rate, use_uncs,
                    train_mc_runs, train_dropout, test_dropout, kernel_size,
                    nhid, lr):
    '''
    Creates the desired prediction model.
    @param mode: which predictor: no, rnn, autoregressive, tfrnn, tftlrnn,tftlrnndense, tcn, kalman;
    no hybrid predictor names allowed (e.g. hybrid-autoregressive-rnn) 
    because they are decomposed in their single predictors before
    @param batch_size: batch size for the RNN
    @param n_time_steps: number of time steps to use for prediction/training
    @param n_features: dimensionality of the solution space
    @param nhid: number of filters
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
    elif mode == "kalman":
        from pykalman import KalmanFilter
        predictor = KalmanFilter(n_dim_obs=n_features, n_dim_state=n_features)
    elif mode == "tfrnn" or mode == "tftlrnn" or mode == "tftlrnndense":
        from utils.utils_transferlearning import build_tl_rnn_predictor
        predictor = build_tl_rnn_predictor(rnn_type, ntllayers,
                                           n_overall_layers, n_time_steps, epochs, n_features,
                                           returnseq, batch_size, apply_tl, with_dense_first,
                                           tl_learn_rate)
    elif mode == "tcn":
        # kernel_size = 3  # 2  # 3
        # 5  # 6  # 5  # 8  # 4
        # #levels like in the paper "an Empirical Evaluation of Generic
        # Convolutional and Reccurent Networks for Sequence Modeling
        levels = math.ceil(math.log(n_time_steps / (2 * (kernel_size - 1)), 2))

        # nhid = 16  # number filters
        in_channels = n_features  # for each dimension one channel
        output_size = n_features  # n_classes
        num_channels = [nhid] * levels  # channel_sizes
        # lr = 0.002  # 2e-4  # learning rate # TODO
        predictor = MyAutoTCN(in_channels, output_size, num_channels,
                              n_time_steps, kernel_size, batch_size,
                              train_mc_runs, train_dropout, test_dropout, lr,
                              init=False, use_uncs=use_uncs)
    elif mode == "truepred":
        predictor = None
    else:
        msg = "unknown prediction mode " + mode
        warnings.warn(msg)
    return predictor


def predict_with_autoregressive(new_train_data, n_features, n_time_steps, scaler, predictor, do_training, n_new_train_data):
    '''
    Predicts next optimum position with autoregressive model.
    @param new_train_data: 
    @param n_new_train_data: 0 if do_training==True

    Throws ValueError (if training data consists data which are nearly zero) 

    http://www.statsmodels.org/devel/tsa.html#univariate-autogressive-processes-ar (overview)
    http://www.statsmodels.org/devel/generated/statsmodels.tsa.ar_model.AR.html#statsmodels.tsa.ar_model.AR (AR)
    http://www.statsmodels.org/devel/vector_ar.html#var (VAR)
    http://www.statsmodels.org/devel/generated/statsmodels.tsa.vector_ar.var_model.VAR.html#statsmodels.tsa.vector_ar.var_model.VAR (VAR)

    https://qiita.com/vinyip918/items/583ac76b9b05cdcc2dde (example for AR)
    '''

    # maximum number of time steps to use for prediction
    # -1 for VAR, otherwise exception (perhaps because otherwise each column has only one value
    # https://datascience.stackexchange.com/questions/38203/var-model-valueerror-x-already-contains-a-constant
    max_lag = n_time_steps - 1
    n_train_data = len(new_train_data)
    if n_features == 1:
        # number of time steps to predict (1 because only next optimum)
        n_prediction_steps = 1

        # TODO implement
        warnings.warn(
            "deprecated: is not adapted to usage of interval training (no usage of predictor object and do_training")
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
        if do_training:
            predictor = VAR(new_train_data)

        # throws exception:  "ValueError: x already contains a constant" if the
        # training data contain values similar to zero
        try:
            # TODO choose one of both lines!!!
            #model_fit = model.fit(maxlags=max_lag)
            model_fit = predictor.fit()
        except ValueError:
            # "ValueError: x already contains a constant"
            # if max_lag = n_time_steps
            model_fit = predictor.fit()
            print("ARR prediction: caught Exception", flush=True)

        lag_order = model_fit.k_ar
        #print('Lag: %s' % lag_order)
        #print('Coefficients: %s' % model_fit.params)

        n_prediction_steps = n_new_train_data + 1

        # make prediction, idea for "forecast" instead of "predict" from here:
        # http://www.statsmodels.org/0.6.1/vector_ar.html (7.10.17)
        # http://www.statsmodels.org/dev/vector_ar.html
        predictions = model_fit.forecast(
            new_train_data[-lag_order:], n_prediction_steps)
        prediction = predictions[-1]

    if n_prediction_steps == 1:
        # make 1d numpy array from 2d array if only one prediction is done
        prediction = prediction.flatten()

    # invert scaling (1d array would result in DeprecatedWarning -> pass 2d)
    converted = scaler.inverse_transform(np.array([prediction]), False)
    return converted.flatten(), predictor


def predict_with_kalman(new_train_data, scaler, predictor,  do_training):
    '''
    Predicts next optimum position with a Kalman filter.
    @param new_train_data: format [n_data, dims]
    '''

    if do_training:
        # "training" of parameters
        predictor.em(new_train_data)

    # computation of states for past observations
    means, covariances = predictor.filter(new_train_data)

    # predicting the next step
    new_measurement = None  # not yet known
    next_mean, next_covariance = predictor.filter_update(
        means[-1], covariances[-1], new_measurement)
    # variance per dimension
    next_variance = np.diagonal(next_covariance)

    # invert scaling (1d array would result in DeprecatedWarning -> pass 2d)
    next_mean = next_mean.reshape(1, -1)
    next_variance = next_variance.reshape(1, -1)
    next_mean = scaler.inverse_transform(next_mean, False).flatten()
    next_variance = scaler.inverse_transform(next_variance, True).flatten()
    assert (next_variance >= 0).all()
    return next_mean, next_variance


def predict_with_rnn(new_train_data, noisy_series, n_epochs, batch_size, n_time_steps,
                     n_features, scaler, predictor, shuffle, pred_np_rnd_generator, do_training):
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

    # separate input series (first values) and prediction value (last
    # value)
    train_in_data, train_out_data = shuffle_split_output(train_samples, False,
                                                         n_time_steps, n_features, shuffle, pred_np_rnd_generator)
    #========================
    # train regressor
    if do_training:
        for i in range(n_epochs):
            hist = predictor.fit(train_in_data, train_out_data, epochs=1,
                                 batch_size=batch_size,  verbose=0, shuffle=shuffle)  # TODO shuffle should be True
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

    n_sampl = 1  # must equal batch size (for RNN)
    reshaped_sample_x = prediction_series.reshape(
        n_sampl, n_time_steps, n_features)
    sample_y_hat = predictor.predict(
        reshaped_sample_x, batch_size=batch_size)

    # invert scaling
    next_optimum = scaler.inverse_transform(sample_y_hat, False).flatten()
    return next_optimum


def predict_with_tfrnn(sess, new_train_data, noisy_series, n_epochs, batch_size, n_time_steps,
                       n_features, scaler, predictor, returnseq, shuffle, pred_np_rnd_generator, do_training):
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
                                                         n_time_steps, n_features, shuffle, pred_np_rnd_generator)
    #========================
    # train regressor
    # TODO save model? report training error?
    if do_training:
        keep_prob = 0.95
        train_error, _, _, train_err_per_epoch, _ = predictor.train(sess, train_in_data, train_out_data, pred_np_rnd_generator, in_keep_prob=keep_prob, out_keep_prob=keep_prob, st_keep_prob=keep_prob,
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
    next_optimum = scaler.inverse_transform(sample_y_hat, False).flatten()
    return next_optimum, train_error, train_err_per_epoch


def predict_with_tcn(sess, new_train_data, noisy_series, n_epochs,
                     n_time_steps, n_features, scaler, predictor,
                     shuffle, do_training,
                     best_found_pos_per_chgperiod, predict_diffs, test_mc_runs,
                     pred_np_rnd_generator):
    '''
    @param do_training: False if model should not be trained but should only
    predict the next step for the given data
    '''

    #========================
    # prepare training data

    # make supervised data from series [#samples,#n_time_steps+1,#n_features]
    train_samples = make_multidim_samples_from_series(
        new_train_data, n_time_steps)
    if noisy_series is not None:
        print("noisy")
        # 4d array [#series, #samples, #n_time_steps+1, #n_features]
        noisy_samples = np.array([make_multidim_samples_from_series(
            noisy_series[i], n_time_steps) for i in range(len(noisy_series))])
        # convert 4d to 3d array [#series*#samples,#n_time_steps+1,#n_features]
        noisy_samples = np.reshape(
            noisy_samples, (-1, n_time_steps + 1, n_features))
        # append noisy samples to "real" samples -> 3d array
        train_samples = np.concatenate((train_samples, noisy_samples))

    # separate input series (first values) and prediction value (last value)
    returnseq = False
    train_in_data, train_out_data = shuffle_split_output(train_samples, returnseq,
                                                         n_time_steps, n_features,
                                                         shuffle, pred_np_rnd_generator)
    n_train = len(train_in_data)
    #========================
    # Training
    #import tensorflow as tf
    log_interval = 1
    #file_writer = tf.summary.FileWriter('./log/train', sess.graph)
    file_writer = None
    if do_training:
        print("train_CNN", flush=True)
        predictor.train(n_epochs, sess, train_in_data, train_out_data,
                        n_train, log_interval, file_writer, shuffle, pred_np_rnd_generator)

    #========================
    # Prediction
    # prediction for next step (with n_time_steps)
    prediction_series = np.array(new_train_data[-n_time_steps:])
    n_sampl = 1  # should be 1
    reshaped_sample_x = prediction_series.reshape(
        n_sampl, n_time_steps, n_features)
    avg_al_unc = None
    if test_mc_runs > 0:
        (pred_var, avg_al_unc,
         pred_mean, predictions) = evaluate_tcn_with_epistemic_unc(sess, predictor, scaler,
                                                                   reshaped_sample_x,
                                                                   test_mc_runs,
                                                                   best_found_pos_per_chgperiod,
                                                                   predict_diffs)
        sample_y_hat = pred_mean
        pred_var = pred_var.flatten()
        avg_al_unc = avg_al_unc.flatten()
    else:
        # this case is executed also if the normal TCN ist trained (without
        # uncertainty estimation)
        sample_y_hat, aleat_unc = predictor.predict(
            sess, reshaped_sample_x, n_sampl, n_features)
        sample_y_hat = scaler.inverse_transform(sample_y_hat, False)
        if predict_diffs:
            sample_y_hat = np.add(
                best_found_pos_per_chgperiod[-1], sample_y_hat)
        pred_var = None
        if aleat_unc is not None:
            avg_al_unc = aleat_unc.flatten()
    # convert 2d-arrays with format [1, n_dims] to 1d arrays with [n_dims]
    next_optimum = sample_y_hat.flatten()

    #========================
    return next_optimum, pred_var, avg_al_unc


def evaluate_tcn_with_epistemic_unc(sess, predictor, scaler,
                                    in_data, test_mc_runs,
                                    best_found_pos_per_chgperiod, pred_diffs):
    '''
    Implements "loss type" proposed by S. Oehmcke (ICANN 2018)

    @return: ep_uncs: 2d array [n_data, n_dims]: predictive variance for each 
    dimension for each input data item
    '''
    # =====================
    # Monte Carlo runs
    predictions = []
    aleat_uncts = []
    for i in range(test_mc_runs):
        print("mc run ", i, flush=True)
        # shape of pred: [len(unsh_in_data), dims]
        # shape of aleat_unc: [len(unsh_in_data), dims or 1]
        (pred, aleat_unc) = evaluate_tcn(sess, predictor, in_data)
        predictions.append(pred)
        aleat_uncts.append(aleat_unc)

    predictions = np.array(predictions)
    aleat_uncts = np.array(aleat_uncts)
    assert (aleat_uncts >= 0).all()

    # =====================
    # re-scale data, transform differences to absolute positions
    predictions, aleat_uncts = rescale_tcn_auto_results(
        scaler, predictions, aleat_uncts, pred_diffs, best_found_pos_per_chgperiod)
    # =====================
    # predictive mean/variance according to S. Oehmckes paper
    pred_mean = np.average(predictions, axis=0)
    avg_squared_preds = np.average(np.square(predictions), axis=0)
    avg_al_uncs = np.average(aleat_uncts, axis=0)
    if len(aleat_uncts.shape) > 2:
        # uncertainties for each dimension
        pass
    else:
        # al. unc. has only one dimension -> add one dimension
        avg_al_uncs = np.array([avg_al_uncs])
        avg_al_uncs = np.transpose(avg_al_uncs)
    pred_var = avg_al_uncs + avg_squared_preds - np.square(pred_mean)

    assert (avg_al_uncs >= 0).all()
    assert (pred_var >= 0).all()
    return pred_var, avg_al_uncs, pred_mean, predictions


def evaluate_tcn(sess, predictor, in_data):
    n_data = len(in_data)
    n_features = in_data.shape[-1]

    total_pred, aleat_unc = predictor.predict(
        sess, in_data, n_data, n_features)

    return total_pred, aleat_unc


def rescale_tcn_auto_results(scaler, predictions, aleat_uncts, pred_diffs, best_found_pos_per_chgperiod):
    for i in range(len(predictions)):
        # format [n_mc_runs, n_data, dims]
        predictions[i] = scaler.inverse_transform(predictions[i], False)
        aleat_uncts[i] = scaler.inverse_transform(aleat_uncts[i], True)
    # redo differences
    if pred_diffs:
        predictions = np.add(best_found_pos_per_chgperiod[-1], predictions)
    return predictions, aleat_uncts


def predict_with_truepred(glob_opt, trueprednoise, pred_np_rnd_generator):
    '''
    Disturbs the known global optimum with known noise that follows a 
    normal distribution with mean=0.0 and standard deviation=trueprednoise
    @param glob_opt: global optimum
    @param trueprednoise: standard deviation of noise 
    @param pred_np_rnd_generator: random number generator
    @return: Tupel (noisy global optimum, variance of noise)  
    '''
    prediction = glob_opt + \
        pred_np_rnd_generator.normal(0, trueprednoise, size=(len(glob_opt)))
    pred_unc = trueprednoise**2
    return prediction, pred_unc


def predict_next_optimum_position(mode, sess, new_train_data, noisy_series, n_epochs, batch_size,
                                  n_time_steps, n_features, scaler, predictor,
                                  returnseq, shuffle, do_training, best_found_pos_per_chgperiod,
                                  predict_diffs, test_mc_runs, n_new_train_data,
                                  glob_opt, trueprednoise, pred_np_rnd_generator):
    '''
    @param mode: the desired predictor
    @param new_train_data: 2d numpy array: contains time series of 
    (n_time_steps+1) previous found solutions (they are already differences if
    predict_diffs is True)
    @param n_epochs: number training epochs for RNN 
    @param batch_size: batch size for RNN
    @param n_time_steps: number of previous solutions to use for the prediction
    @param n_features: dimensionality of the solution space
    @param scaler: scaler to re-scale the prediction (because the training data 
    is already scaled)
    @param predictor: RNN predictor object to use for the prediction, so that
    it is not required to train it completely new
    @param n_new_train_data: 0 if do_training==True
    @return 1d numpy array: position of next optimum (already re-scaled)
    '''
    train_error = None
    train_err_per_epoch = None
    pred_unc = None
    avg_al_unc = None
    ar_predictor = None
    if mode == "rnn":
        prediction = predict_with_rnn(new_train_data, noisy_series, n_epochs, batch_size,
                                      n_time_steps, n_features, scaler, predictor, shuffle,
                                      pred_np_rnd_generator, do_training)
    elif mode == "autoregressive":
        try:
            prediction, ar_predictor = predict_with_autoregressive(
                new_train_data, n_features, n_time_steps, scaler, predictor, do_training, n_new_train_data)
        except ValueError:
            raise
    elif mode == "kalman":
        prediction, pred_unc = predict_with_kalman(
            new_train_data, scaler, predictor, do_training)
    elif mode == "no":
        prediction = None
    elif mode == "tfrnn" or mode == "tftlrnn" or mode == "tftlrnndense":
        prediction, train_error, train_err_per_epoch = predict_with_tfrnn(sess, new_train_data, noisy_series, n_epochs, batch_size, n_time_steps,
                                                                          n_features, scaler, predictor, returnseq, shuffle,
                                                                          pred_np_rnd_generator, do_training)
    elif mode == "tcn":
        prediction, pred_unc, avg_al_unc = predict_with_tcn(sess, new_train_data, noisy_series, n_epochs,
                                                            n_time_steps, n_features, scaler, predictor,
                                                            shuffle, do_training,
                                                            best_found_pos_per_chgperiod, predict_diffs,
                                                            test_mc_runs, pred_np_rnd_generator)
    elif mode == "truepred":
        prediction, pred_unc = predict_with_truepred(
            glob_opt, trueprednoise, pred_np_rnd_generator)

    # convert predicted difference into position (tcn has already re-scaled and
    # added the values in the sub-functions)
    # TODO (dev) add possibly predictor type
    if predict_diffs and mode != "tcn" and mode != "truepred":
        prediction = np.add(best_found_pos_per_chgperiod[-1], prediction)
    return prediction, train_error, train_err_per_epoch, pred_unc, avg_al_unc, ar_predictor


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


def fit_scaler(data_for_fitting):
    scaler = MyMinMaxScaler(feature_range=(-1, 1))
    scaler.fit(data_for_fitting)
    return scaler


def get_noisy_time_series(original_series, n_series, stddev_per_chgperiod, pred_np_rnd_generator):
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
    return pred_np_rnd_generator.normal(loc=original_series, scale=stddev_per_chgperiod,
                                        size=(n_series, n_chgperiods, dims))


def get_first_chgp_idx_with_pred(overall_n_chgperiods, n_preds):
    '''
    Computes index of first change period for that a prediction was made

    @param overall_n_chgperiods: number of change periods for that the EA was run
    @param n_preds: number of change periods for that a prediction was made
    @return scalar
    '''
    return np.arange(overall_n_chgperiods)[-n_preds]


def get_first_generation_idx_with_pred(overall_n_chgperiods, n_preds,
                                       gens_of_chgperiods):
    '''
    Computes index of first generation of the first change period for that a 
    prediction was made.

    @param overall_n_chgperiods: number of change periods for that the EA was run
    @param n_preds: number of change periods for that a prediction was made
    @param gens_of_chgperiods: dictionary: for each change period (even if the
    EA did not detect it) a list of the corresponding generations
    @return scalar
    '''
    first_chpg_idx = get_first_chgp_idx_with_pred(
        overall_n_chgperiods, n_preds)

    # first generation of first change period with prediction
    return gens_of_chgperiods[first_chpg_idx][0]


def calculate_n_train_samples(n_past_chgps, pred_diffs, n_time_steps):
    '''
    Calculates the number of training samples (input&output) that could be
    produced from the finished change periods.
    Inverse to calculate_n_required_chgps_from_n_train_samples()

    @param n_past_chgps: number of change periods for that the EA already has
    found a solution.
    @param pred_diffs: True if not absolute positions but differences are predicted
    @param n_time_steps: number of steps that are used to predict the next
    @return scalar
    '''
    if pred_diffs:
        return n_past_chgps - 1 - n_time_steps
    else:
        return n_past_chgps - n_time_steps


def calculate_n_required_chgps_from_n_train_samples(n_train_samples, pred_diffs, n_time_steps):
    '''
    Calculates the number of finished change periods required to produce the 
    desired number of training samples (input&output). 
    Inverse to calculate_n_train_samples()

    @param n_train_samples: required number of training samples
    @param pred_diffs: True if not absolute positions but differences are predicted
    @param n_time_steps: number of steps that are used to predict the next
    @return scalar
    '''
    if pred_diffs:
        return n_train_samples + n_time_steps + 1
    else:
        return n_train_samples + n_time_steps


def prepare_data_train_and_predict(sess, gen_idx, n_features, predictors,
                                   experiment_data, n_epochs, batch_size,
                                   return_seq, shuffle_train_data, n_new_train_data,
                                   best_found_pos_per_chgperiod, train_interval,
                                   predict_diffs, n_time_steps, n_required_train_data,
                                   predictor_name, add_noisy_train_data,
                                   n_noisy_series, stddev_among_runs_per_chgp,
                                   test_mc_runs, benchmarkfunction, use_uncs,
                                   pred_unc_per_chgperiod, aleat_unc_per_chgperiod,
                                   pred_opt_pos_per_chgperiod, pred_opt_fit_per_chgperiod,
                                   train_error_per_chgperiod,
                                   train_error_for_epochs_per_chgperiod,
                                   glob_opt, trueprednoise, pred_np_rnd_generator):
    '''
    TODO use this function in dynpso
    '''
    n_past_chgps = len(best_found_pos_per_chgperiod)
    # number of train data that can be produced from the last chg. periods
    overall_n_train_data = calculate_n_train_samples(
        n_past_chgps, predict_diffs, n_time_steps)

    # prevent training with too few train data
    if (overall_n_train_data < n_required_train_data or predictor_name == "no"):
        my_pred_mode = "no"
        train_data = None
        prediction = None

    else:
        my_pred_mode = predictor_name

        # number of required change periods (to construct training data)
        n_required_chgps = calculate_n_required_chgps_from_n_train_samples(
            n_required_train_data, predict_diffs, n_time_steps)
        best_found_vals_per_chgperiod = best_found_pos_per_chgperiod[-n_required_chgps:]

        # transform absolute values to differences
        if predict_diffs:
            best_found_vals_per_chgperiod = np.subtract(
                best_found_vals_per_chgperiod[1:], best_found_vals_per_chgperiod[:-1])

        # scale data (the data are re-scaled directly after the
        # prediction in this iteration)
        scaler = fit_scaler(best_found_vals_per_chgperiod)
        train_data = scaler.transform(
            copy.copy(best_found_vals_per_chgperiod))

        # add noisy training data in order to make network more robust and
        # increase the number of training data
        if add_noisy_train_data:
            # 3d array [n_series, n_chgperiods, dims]
            noisy_series = get_noisy_time_series(np.array(best_found_pos_per_chgperiod),
                                                 n_noisy_series,
                                                 stddev_among_runs_per_chgp, pred_np_rnd_generator)
            if predict_diffs:
                noisy_series = np.array([np.subtract(
                    noisy_series[i, 1:], noisy_series[i, :-1]) for i in range(len(noisy_series))])
            # scale data
            noisy_series = np.array([scaler.transform(
                copy.copy(noisy_series[i])) for i in range(len(noisy_series))])
        else:
            noisy_series = None

        # train data
        train_data = np.array(train_data)
        # train the model only when train_interval new data are available
        do_training = n_new_train_data >= train_interval
        if do_training:
            n_new_train_data = 0
        # predict next optimum position or difference (and re-scale value)

        prdctns = []
        prdctns_fit = []
        pred_unc_per_predictor = []
        avg_al_unc_per_predictor = []
        train_error_per_predictor = []
        train_err_per_epoch_per_predictor = []
        prdct_nms = []
        for prdctr_name in predictors.keys():
            # make prediction with each single predictor
            (prdctn, train_error, train_err_per_epoch,
             pred_unc, avg_al_unc, ar_predictor) = predict_next_optimum_position(prdctr_name, sess, train_data, noisy_series,
                                                                                 n_epochs, batch_size,
                                                                                 n_time_steps, n_features,
                                                                                 scaler, predictors[
                                                                                     prdctr_name], return_seq, shuffle_train_data,
                                                                                 do_training, best_found_pos_per_chgperiod,
                                                                                 predict_diffs, test_mc_runs, n_new_train_data,
                                                                                 glob_opt, trueprednoise, pred_np_rnd_generator)
            prdctns.append(prdctn)
            prdctns_fit.append(utils_dynopt.fitness(
                benchmarkfunction, prdctn, gen_idx, experiment_data))
            pred_unc_per_predictor.append(pred_unc)
            avg_al_unc_per_predictor.append(avg_al_unc)
            train_error_per_predictor.append(train_error)
            train_err_per_epoch_per_predictor.append(train_err_per_epoch)
            prdct_nms.append(prdctr_name)
            if prdctr_name == "autoregressive":
                predictors[prdctr_name] = ar_predictor

        # index of best prediction
        min_idx = np.argmin(prdctns_fit)
        prediction = prdctns[min_idx]
        prediction_fit = prdctns_fit[min_idx]
        pred_unc = pred_unc_per_predictor[min_idx]
        avg_al_unc = avg_al_unc_per_predictor[min_idx]
        train_error = train_error_per_predictor[min_idx]
        train_err_per_epoch = train_err_per_epoch_per_predictor[min_idx]

        # store results
        pred_opt_pos_per_chgperiod.append(copy.copy(prediction))
        pred_opt_fit_per_chgperiod.append(prediction_fit)
        if pred_unc is not None and use_uncs:
            pred_unc_per_chgperiod.append(copy.copy(pred_unc))
        if avg_al_unc is not None and use_uncs:
            aleat_unc_per_chgperiod.append(copy.copy(avg_al_unc))
        train_error_per_chgperiod.append(train_error)
        train_error_for_epochs_per_chgperiod.append(
            train_err_per_epoch)
    return my_pred_mode, predictors, n_new_train_data
