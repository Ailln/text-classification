import tensorflow as tf


class Model(object):
    def __init__(self, config):
        self.max_len = config["input_max_len"]
        self.embedding_size = config["embedding_size"]
        self.num_classes = config["num_classes"]
        self.filter_size = config["filter_size"]
        self.kernel_size = config["kernel_size"]
        self.opt_name = config["opt_name"]
        self.lr = config["learning_rate"]
        self.input_holder = tf.placeholder(tf.float32, [None, self.max_len, self.embedding_size], "input_data")
        self.target_holder = tf.placeholder(tf.int32, [None, self.num_classes], "target_data")

    def cnn(self):
        # conv1d: inputs, filters, kernel_size
        # conv: [batch_size, max_len, filters]
        conv_output = tf.layers.conv1d(self.input_holder, self.filter_size, self.kernel_size, name="conv1d")
        # gmp: [batch_size, filters]
        gmp = tf.reduce_max(conv_output, reduction_indices=[1], name='gmp')
        # logits: [batch_size, num_classes]
        logits = tf.layers.dense(gmp, self.num_classes, name="fc")

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.target_holder)

        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss', loss)
        if self.opt_name == "adam":
            train_step = tf.train.AdamOptimizer(self.lr).minimize(loss)
        else:
            train_step = tf.train.AdagradOptimizer(self.lr).minimize(loss)

        pred = tf.argmax(tf.nn.softmax(logits), 1)
        return train_step, pred, loss

