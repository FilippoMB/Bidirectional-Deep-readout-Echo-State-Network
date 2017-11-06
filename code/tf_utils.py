import numpy as np
import tensorflow as tf
import time
#from tqdm import tqdm, trange


def next_batch(X, Y, batch_size=1, shuffle=True):
    # Generator that supplies mini batches
    n_data = len(Y)

    if shuffle:
        idx = np.random.permutation(n_data)
    else:
        idx = range(n_data)

    # Timeseries or vectorial data?
    if len(X.shape) == 3:
        X = X[:, idx, :]
    else:
        X = X[idx, :]

    Y = Y[idx, :]

    n_batches = n_data//batch_size

    for i in range(n_batches):
        if len(X.shape) == 3:
            X_batch = X[:, i*batch_size:(i + 1)*batch_size, :]
        else:
            X_batch = X[i*batch_size:(i + 1)*batch_size, :]

        Y_batch = Y[i*batch_size:(i + 1)*batch_size]

        yield X_batch, Y_batch


def train_tf_model(model_name, graph, X, Y, Xte, Yte, n_data, batch_size,
                   num_epochs, nn_input, keep_prob, logits, nn_output,
                   w_l2, learning_rate, p_drop):

    with graph.as_default():
        with tf.Session(graph=graph) as sess:
            min_val_loss = np.infty
            model_name = 'models/' + model_name + '_0.ckpt'
            saver = tf.train.Saver()

            # Define loss
            class_loss = \
                tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=logits,
                        labels=nn_output
                        )
                    )

            # L2 loss
            reg_loss = 0
            for tf_var in tf.trainable_variables():
                reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

            # Define optimizer
            tot_loss = class_loss + w_l2*reg_loss
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(tot_loss)

            # Evaluate model (with test logits, for dropout to be disabled)
            pred_class = tf.argmax(logits, axis=1)
            true_class = tf.argmax(nn_output, axis=1)
            correct_pred = tf.equal(pred_class, true_class)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            sess.run(tf.global_variables_initializer())

            # ================= TRAINING =================

            # initialize training stuff
            time_tr_start = time.time()
            loss_track = []

            try:
#                for t in trange(num_epochs, desc='\tComplete', unit=" epochs", ncols=70):
                for t in range(num_epochs):

                    for X_batch, Y_batch in next_batch(X=X,
                                                       Y=Y,
                                                       batch_size=batch_size,
                                                       shuffle=True):
                        fdtr = {nn_input: X_batch,
                                nn_output: Y_batch,
                                keep_prob: p_drop}

                        _, train_loss = sess.run([train_op, tot_loss], fdtr)

                        loss_track.append(train_loss)

                    # check training progress on the whole set
                    if t % 100 == 0:
                        fdvs = {nn_input: X,
                                nn_output: Y,
                                keep_prob: 1.0}
                        val_acc, val_loss = sess.run([accuracy, tot_loss], fdvs)  # summary, merged_summary
                        # train_writer.add_summary(summary, ep)
                        #tqdm.write('\t\t VS acc=%.3f, loss=%.3f -- TR loss=%.3f' % (val_acc, val_loss, np.mean(loss_track[-100:])))

                        # Save model yielding best results on validation
                        if val_loss < min_val_loss:
                            min_val_loss = val_loss
                            tf.add_to_collection("nn_input", nn_input)
                            tf.add_to_collection("nn_output", nn_output)
                            tf.add_to_collection("accuracy", accuracy)
                            tf.add_to_collection("tot_loss", tot_loss)
                            tf.add_to_collection("pred_class", pred_class)
                            save_path = saver.save(sess, model_name)

            except KeyboardInterrupt:
                print('training interrupted')

            saver.restore(sess, model_name)

            ttime = (time.time()-time_tr_start)/60

            fdte = {nn_input: Xte,
                    nn_output: Yte,
                    keep_prob: 1.0}
            te_pred_class, te_accuracy, te_loss = sess.run([pred_class, accuracy, tot_loss], fdte)


    return loss_track, te_pred_class, ttime
