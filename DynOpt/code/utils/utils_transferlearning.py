'''
Created on Nov 17, 2018

@author: ameier
'''
import math
import re


def get_variables_and_names(ntllayers):
    import tensorflow as tf
    '''
    Returns three different types of variables and their names:
        - the pre-trained weights and bias of transfer learning layers
        - the variables of the new layers after the transfer learning layers
          (recurrent and dense output layers)
        - all trainable variables both in tl and prediction layers

    TODO funktioniert das auch ohne Transfer Learning?
    '''

    allow_print = False
    # variable names for weights/bias have the following pattern, e.g:
    # rnn/multi_rnn_cell/cell_0/lstm_cell/bias:0
    # rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0
    # https://stackoverflow.com/questions/45617026/tensorflow-transfer-learning-how-to-load-part-of-layers-from-one-checkpoint-file
    # (7.11.18)
    pattern = re.compile(
        "/multi_rnn_cell/cell_[0-" + str(ntllayers - 1) + "]")

    # pre-trained variables
    tl_variables = [var for var in tf.global_variables()
                    if pattern.search(var.name) and (var.name.endswith('/bias:0')
                                                     or var.name.endswith('/kernel:0'))]
    tl_vars_names = [var.name for var in tl_variables]
    if allow_print:
        print("\n\n", flush=True)
        print("variables to restore: ", flush=True)
        for name in tl_vars_names:
            print(name, flush=True)

    # all trainable variables
    all_variables = tf.trainable_variables()
    all_vars_names = [var.name for var in all_variables]
    if allow_print:
        print("all variables: ", flush=True)
        for name in all_vars_names:
            print(name, flush=True)

    # the variables that should be trained (the new prediction layers, but
    # not the pre-trained tl layers)
    predlayer_variables = []
    predlayer_vars_names = []
    for var, name in zip(all_variables, all_vars_names):
        if name not in tl_vars_names:
            predlayer_variables.append(var)
            predlayer_vars_names.append(name)
    if allow_print:
        print("predlayer_vars_names: ", flush=True)
        for name in predlayer_vars_names:
            print(name, flush=True)

    return tl_variables, tl_vars_names, predlayer_variables, \
        predlayer_vars_names,  all_variables, all_vars_names


def build_tl_model(rnn_type, ntllayers, ntimesteps,
                   epochs, dims, returnseq, test_b_size, with_dense_first):
    from utils.tensorflow_rnnclass_withoutstate import TFRNNWithoutState
    # build transfer learning network
    return TFRNNWithoutState(dims, n_time_steps_to_use=ntimesteps,
                             test_b_size=test_b_size, n_epochs=epochs, has_time_outputs=returnseq,
                             custom_reset=False, n_rnn_layers=ntllayers, n_neurons_per_layer=None,
                             rnn_type=rnn_type, with_dense_first=with_dense_first)


def build_tl_rnn_predictor(rnn_type, ntllayers, n_overall_layers,
                           ntimesteps, epochs, dims,
                           returnseq, test_b_size, apply_tl, with_dense_first,
                           tl_learn_rate):
    '''
    Builds the prediction model.

    In case transfer learning is applied, the weights and biases for the tl layers
    (except those for the dense output layer) are reused. An additional
    rnn (or lstm) layer is added after the tl layers and a new dense output 
    layer is created. The overall number of recurrent layers: ntllayers+npredlayers

    In case transfer learning is not applied, a new untrained network is created.
    Overall number of recurrent layers: npredlayers

    @return: prediction model (a tensorflow model)
    '''
    import tensorflow as tf
    # build completely new network with all layers
    pred_model = build_tl_model(rnn_type, n_overall_layers, ntimesteps,
                                epochs, dims, returnseq, test_b_size, with_dense_first)

    if apply_tl:
        # set learning rates for tl and prediction layers separately (8.11.18)
        # https://stackoverflow.com/questions/34945554/how-to-set-layer-wise-learning-rate-in-tensorflow

        # default learning rate for AdamOptimizer in Tensorflow: 0.001
        # https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
        tl_variables, _, predlayer_variables, _, _, _ = get_variables_and_names(
            ntllayers)
        var_list1 = tl_variables  # first layers
        var_list2 = predlayer_variables  # remaining layers
        opt1 = tf.train.AdamOptimizer(tl_learn_rate)
        opt2 = tf.train.AdamOptimizer(0.001)
        grads = tf.gradients(pred_model.loss, var_list1 + var_list2)
        grads1 = grads[:len(var_list1)]
        grads2 = grads[len(var_list1):]
        train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
        train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
        pred_model.train_op = tf.group(train_op1, train_op2)  # new train_op
    return pred_model
