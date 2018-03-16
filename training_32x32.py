import numpy as np
import tensorflow as tf
import math
import sys
import time
import datetime
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class NISTHelper():
    def __init__(self, train_img, train_label, test_img, test_label):
        self.i = 0
        self.test_i = 0
        self.training_images = train_img
        self.training_labels = train_label
        self.test_images = test_img
        self.test_labels = test_label

    def next_batch(self, batch_size):
        x = self.training_images[self.i:self.i + batch_size]
        y = self.training_labels[self.i:self.i + batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y

    def test_batch(self, batch_size):
        x = self.test_images[self.test_i:self.test_i + batch_size]
        y = self.test_labels[self.test_i:self.test_i + batch_size]
        self.test_i = (self.test_i + batch_size) % len(self.test_images)
        return x, y


def unison_shuffled_copies(a, b):
    """Returns 2 unison shuffled copies of array a and b"""
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# HELPER FUNCTIONS
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def init_weights(shape):
    """Returns random initial weights"""
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)


def init_bias(shape):
    """Returns random initial biases"""
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)


def conv2d(x, W):
    """Returns a 2d convolution operation with stride size 1 and padding SAME"""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2by2(x):
    """Returns a 2 by 2 pooling operation with padding SAME"""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def convolutional_layer(input_x, shape, name="unspecified"):
    """Returns a convolutional layer with random weights and biases"""
    with tf.name_scope(name):
        with tf.name_scope("weights"):
            W = init_weights(shape)
            variable_summaries(W)
        with tf.name_scope("biases"):
            b = init_bias([shape[3]])
        with tf.name_scope("Wx_plus_b"):
            preactive = conv2d(input_x, W) + b
            tf.summary.histogram("pre_activations", preactive)
    activations = tf.nn.relu(preactive, name="activation")
    tf.summary.histogram("activations", activations)
    return activations


def normal_full_layer(input_layer, size, act=tf.nn.relu, name="unspecified"):
    """Returns a full layer with random weights and biases"""
    with tf.name_scope(name):
        input_size = int(input_layer.get_shape()[1])
        with tf.name_scope("weights"):
            W = init_weights([input_size, size])
            variable_summaries(W)
        with tf.name_scope("biases"):
            b = init_bias([size])
            variable_summaries(b)
        with tf.name_scope("Wx_plus_b"):
            preactive = tf.matmul(input_layer, W) + b
            tf.summary.histogram("pre_activations", preactive)
        activations = act(preactive, name="activation")
        tf.summary.histogram("activations", activations)
        return activations


def log(logstr):
    """Prints logstr to console with current time"""
    print(datetime.datetime.now().isoformat() + " " + logstr)


def main():
    # LOADING DATA
    log("Loading data...")
    images = np.load("nist_images_32x32.npy")
    labels = np.load("nist_labels_32x32.npy")
    log("Data loaded... Shuffling...")
    images, labels = unison_shuffled_copies(images, labels)
    log("Shuffled!")
    split = math.ceil(len(images) * 0.7)
    train_imgs = images[:split]
    train_labels = labels[:split]
    test_imgs = images[split:]
    test_labels = labels[split:]
    log("Performed train-test split")
    nist = NISTHelper(train_imgs, train_labels, test_imgs, test_labels)

    # VARIABLES
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 1], name="x")  # Input, shape = ?x32x32x1
    y_true = tf.placeholder(tf.float32, shape=[None, 47], name="y_true")  # Labels

    # MODEL
    # filter size=(4,4); channels=1; filters=16; shape=?x32x32x32
    convo_1 = convolutional_layer(x, shape=[4, 4, 1, 32], name="Convolutional_1")
    convo_1_pooling = max_pool_2by2(convo_1)  # shape=?x16x16x32

    # filter size=(4,4); channels=16; filters=32; shape=?x16x16x64
    convo_2 = convolutional_layer(convo_1_pooling, shape=[4, 4, 32, 64], name="Convolutional_2")
    convo_2_pooling = max_pool_2by2(convo_2)  # shape=?x8x8x64
    convo_2_flat = tf.reshape(convo_2_pooling, [-1, 8*8*64])
    
    # filter size=(4,4); channels=32; filters=64; shape=?x8x8x32
    #convo_3 = convolutional_layer(convo_2_pooling, shape=[4, 4, 32, 64], name="Convolutional_3")
    #convo_3_pooling = max_pool_2by2(convo_3)  # shape=4x4x32
    #convo_3_flat = tf.reshape(convo_3_pooling, [-1, 4 * 4 * 64])  # Flatten convolutional layer

    full_layer_one = normal_full_layer(convo_2_flat, 1024, tf.nn.relu, name="Normal_Layer_1")
    with tf.name_scope("dropout"):
        hold_prob = tf.placeholder(tf.float32)
        tf.summary.scalar("dropout_keep_probability", hold_prob)
        full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)
    y_pred = normal_full_layer(full_one_dropout, 47, act=tf.identity,
                               name="Output_Layer")  # Layer with 47 neurons for one-hot encoding
    with tf.name_scope("cross_entropy"):
        with tf.name_scope("total"):
            # Calculate cross-entropy
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred))
    tf.summary.scalar("cross_entropy", cross_entropy)
    with tf.name_scope("train"):
        train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)
    with tf.name_scope("accuracy"):
        with tf.name_scope("correct_predictions"):
            correct_predictions = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))  # use argmax to get the index
            # of the highest value in the prediction array and compare that with the true array to generate and array
            #  of the form [True,False,True]
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))  # Calculate percentage of correct
            # predictions
    tf.summary.scalar("accuracy", accuracy)
    log("Model created!")
    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    epoch_start = 0
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter("log/train", sess.graph)
        test_writer = tf.summary.FileWriter("log/test")
        sess.run(init)
        log("Variables initialized!")
        epochs = int(sys.argv[1])
        log("Training for " + str(epochs) + " epochs.")
        for i in range(epochs):
            batch = nist.next_batch(100)
            # Use a hold probability of 0.5 to prevent overfitting
            summary, _ = sess.run([merged, train], feed_dict={x: batch[0], y_true: batch[1], hold_prob: 0.5})
            train_writer.add_summary(summary, i)  # Write epoch to summary
            if (i % 200) == 0:  # Every 200 epochs evaluate with test set
                matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
                acc = tf.reduce_mean(tf.cast(matches, tf.float32))
                batch = nist.test_batch(200)
                # Hold probability is 1 to get the best results
                summary, accuracy = sess.run([merged, acc], feed_dict={x: batch[0], y_true: batch[1], hold_prob: 1})
                test_writer.add_summary(summary, i)  # Save the results of test batch
                saver.save(sess, "log/model.ckpt", i)  # Save model
                if i > 0:
                    log("Step: " + str(i) + "; Accuracy: " + str(accuracy) + "; Time (200 Steps): " + str((time.time() - epoch_start)))
                else:
                    log("Step: " + str(i) + "; Accuracy: " + str(accuracy) + ";")
                epoch_start = time.time()
        log("Finished training.")
        model_path = "models/32x32_2conv_32_64_1norm_1024.ckpt"
        saver.save(sess, model_path)  # Save final model
        log("Model saved in " + model_path)


if __name__ == "__main__":
    main()
