import tensorflow as tf


class Model(object):
    def __init__(self, config):
        self.seq_length = config["seq_length"]
        self.embedding_size = config["embedding_size"]
        self.vocab_size = config["vocab_size"]
        self.num_classes = config["num_classes"]
        self.filter_size = config["filter_size"]
        self.kernel_size = config["kernel_size"]
        self.opt_name = config["opt_name"]
        self.lr = config["learning_rate"]
        self.dropout_keep_prob = config["dropout_keep_prob"]
        self.input_holder = tf.placeholder(tf.int32, [None, self.seq_length], "input_data")
        self.target_holder = tf.placeholder(tf.int32, [None, self.num_classes], "target_data")

    def cnn(self):
        # Embedding layer
        with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
            embedding = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                name="embedding"
            )
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_holder)

        # conv1d: inputs, filters, kernel_size
        # conv: [batch_size, seq_length, filters]
        conv_output = tf.layers.conv1d(embedding_inputs, self.filter_size, self.kernel_size, name="conv1d")
        # gmp: [batch_size, filters]
        gmp = tf.reduce_max(conv_output, reduction_indices=[1], name='gmp')
        # logits: [batch_size, num_classes]
        logits = tf.layers.dense(gmp, self.num_classes, name="fc")

        # dropout layer
        # 防止过拟合
        with tf.name_scope("dropout_layer"):
            dropout_logits = tf.nn.dropout(logits, self.dropout_keep_prob)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=dropout_logits, labels=self.target_holder)

        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss', loss)
        if self.opt_name == "adam":
            train_step = tf.train.AdamOptimizer(self.lr).minimize(loss)
        else:
            train_step = tf.train.AdagradOptimizer(self.lr).minimize(loss)

        pred = tf.argmax(tf.nn.softmax(logits), 1)
        return train_step, pred, loss

