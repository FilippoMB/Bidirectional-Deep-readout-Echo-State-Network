# General imports
import math
import numpy as np
import tensorflow as tf
import time
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score

# Custom imports
from tf_utils import train_tf_model
from reservoir import Reservoir


def fc_layer(input_, in_dim, size):
    W = tf.Variable(
            tf.random_normal(
                shape=(in_dim, size),
                stddev=math.sqrt(4.0 / (in_dim + size)),
                ),
            name='weights'
            )

    b = tf.Variable(tf.zeros([size]), name='biases')

    with tf.name_scope('Wx_plus_b'):
        result = tf.add(tf.matmul(input_, W), b)

    return result


def build_network(input_, in_dim, layout, n_classes, keep_prob):
    for i, neurons in enumerate(layout):
        with tf.name_scope('h{}'.format(i)):
            layer = fc_layer(input_, in_dim, neurons)
            layer = tf.nn.relu(layer)
            layer = tf.nn.dropout(layer, keep_prob=keep_prob)

            input_ = layer
            in_dim = neurons

    with tf.name_scope('out'):
        layer = fc_layer(input_, in_dim, n_classes)
        logits = tf.nn.relu(layer)

    return logits


def train_ESN(X, Y,
              Xte, Yte,
              embedding_method,
              n_dim,
              w_ridge,
              n_internal_units=None,
              spectral_radius=None,
              connectivity=None,
              input_scaling=None,
              noise_level=None,
              reservoir=None):
    
    num_classes = Y.shape[1]

    # Initialize reservoir
    if reservoir is None:
        reservoir = Reservoir(n_internal_units, spectral_radius, connectivity,
                              input_scaling, noise_level)
    elif n_internal_units is None \
            or spectral_radius is None \
            or connectivity is None \
            or input_scaling is None \
            or noise_level is None:
        raise RuntimeError('Reservoir parameters missing')

    # Initialize timer
    time_tr_start = time.time()

    # Compute reservoir states
    res_states = reservoir.get_states(X, embedding=embedding_method,
                                      n_dim=n_dim, train=True, bidir=False)

    # Readout training
    readout = Ridge(alpha=w_ridge)
    readout.fit(res_states, Y)

    training_time = (time.time()-time_tr_start)/60

    # Test
    res_states_te = reservoir.get_states(Xte, embedding=embedding_method,
                                         n_dim=n_dim, train=False,
                                         bidir=False)
    logits = readout.predict(res_states_te)
    pred_class = np.argmax(logits, axis=1)
    true_class = np.argmax(Yte, axis=1)
    
    accuracy = accuracy_score(true_class, pred_class)
    if num_classes > 2:
        f1 = f1_score(true_class, pred_class, average='weighted')
    else:
        f1 = f1_score(true_class, pred_class, average='binary')

    return accuracy, f1, training_time


def train_RNN(X,
              Y,
              Xte,
              Yte,
              num_cells,
              fc_layout,
              batch_size,
              num_epochs,
              p_drop,
              w_l2,
              learning_rate,
              cell_type='GRU',
              seed=None):

    num_classes = Y.shape[1]

    # Transposing (time-major)
    X = np.transpose(X, (1, 0, 2))
    Xte = np.transpose(Xte, (1, 0, 2))
    _, n_data, input_size = X.shape

    graph = tf.Graph()

    with graph.as_default():
        if seed is not None:
            tf.set_random_seed(seed)

        # tf Graph input
        nn_input = tf.placeholder(shape=(None, None, input_size),
                                  dtype=tf.float32)
        nn_output = tf.placeholder(shape=(None, num_classes),
                                   dtype=tf.float32)
        keep_prob = tf.placeholder(dtype=tf.float32)

        # ====== RNN =======
        if cell_type == 'RNN':
            rnn_cell = tf.contrib.rnn.BasicRNNCell(num_cells)
        elif cell_type == 'LSTM':
            rnn_cell = tf.contrib.rnn.BasicRNNCell(num_cells)
        elif cell_type == 'GRU':
            rnn_cell = tf.contrib.rnn.GRUCell(num_cells)

        (_, last_state) = (tf.nn.dynamic_rnn(
                rnn_cell,
                inputs=nn_input,
                sequence_length=None,
                time_major=True,
                dtype=tf.float32))
        
        if isinstance(last_state, tf.contrib.rnn.LSTMStateTuple): 
            last_state = last_state.h

        # ======= MLP =======
        logits = build_network(
            last_state,
            num_cells,
            fc_layout,
            num_classes,
            keep_prob)

    loss_track, pred_class, training_time = \
        train_tf_model(cell_type, graph, X, Y, Xte, Yte, n_data,
                       batch_size, num_epochs, nn_input, keep_prob,
                       logits, nn_output, w_l2, learning_rate,
                       p_drop)
        
    true_class = np.argmax(Yte, axis=1)    
    accuracy = accuracy_score(true_class, pred_class)
    if num_classes > 2:
        f1 = f1_score(true_class, pred_class, average='weighted')
    else:
        f1 = f1_score(true_class, pred_class, average='binary')
    
    return loss_track, accuracy, f1, training_time


def train_BDESN(X,
                Y,
                Xte,
                Yte,
                embedding_method,
                n_dim,
                fc_layout,
                batch_size,
                num_epochs,
                p_drop,
                w_l2,
                learning_rate,
                seed=None,
                n_internal_units=None,
                spectral_radius=None,
                connectivity=None,
                input_scaling=None,
                noise_level=None,
                reservoir=None):

    num_classes = Y.shape[1]

    # Compute reservoir states
    if reservoir is None:
        reservoir = Reservoir(n_internal_units, spectral_radius, connectivity,
                              input_scaling, noise_level)
    elif n_internal_units is None \
            or spectral_radius is None \
            or connectivity is None \
            or input_scaling is None \
            or noise_level is None:
        raise RuntimeError('Reservoir parameters missing')

    res_states = reservoir.get_states(X, embedding=embedding_method,
                                      n_dim=n_dim, train=True, bidir=True)

    res_states_te = reservoir.get_states(Xte, embedding=embedding_method,
                                         n_dim=n_dim, train=False,
                                         bidir=True)

    n_data, input_size = res_states.shape

    graph = tf.Graph()
    with graph.as_default():
        if seed is not None:
            tf.set_random_seed(seed)

        # ============= MLP ==============
        # tf Graph input
        nn_input = tf.placeholder(shape=(None, input_size),
                                  dtype=tf.float32)
        nn_output = tf.placeholder(shape=(None, num_classes),
                                   dtype=tf.float32)
        keep_prob = tf.placeholder(dtype=tf.float32)

        # MLP
        logits = build_network(
            nn_input,
            input_size,
            fc_layout,
            num_classes,
            keep_prob)

    loss_track, pred_class, training_time = \
        train_tf_model('BDESN', graph, res_states, Y, res_states_te,
                       Yte, n_data, batch_size, num_epochs, nn_input,
                       keep_prob, logits, nn_output, w_l2,
                       learning_rate, p_drop)

    true_class = np.argmax(Yte, axis=1)    
    accuracy = accuracy_score(true_class, pred_class)
    if num_classes > 2:
        f1 = f1_score(true_class, pred_class, average='weighted')
    else:
        f1 = f1_score(true_class, pred_class, average='binary')
    
    return loss_track, accuracy, f1, training_time
