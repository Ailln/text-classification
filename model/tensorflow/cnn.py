import tensorflow as tf


class Model(object):
    def __init__(self, config):
        self.max_len = config["input_max_len"]
        self.input_holder = tf.placeholder(tf.float32, [None, self.max_len, 1], "input_data")
        self.target_holder = tf.placeholder(tf.int32, [None, 5], "target_data")

    def cnn(self):
        conv = tf.layers.conv1d(self.input_holder, 32, 3, name="conv1d")
        gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
        logits = tf.layers.dense(gmp, 5, name="fc")
        pred = tf.argmax(tf.nn.softmax(logits), 1)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.target_holder)
        loss = tf.reduce_mean(cross_entropy)
        opt = tf.train.AdamOptimizer(0.0001)
        train_step = opt.minimize(loss)
        return train_step, pred, loss

