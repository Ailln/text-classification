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
        self.input_holder = tf.placeholder(tf.int32, [None, self.seq_length], "input_data")
        self.target_holder = tf.placeholder(tf.int32, [None, self.num_classes], "target_data")

    def rnn(self):
        pass
