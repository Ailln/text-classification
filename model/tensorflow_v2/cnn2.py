from tensorflow import keras
from tensorflow.keras import layers


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

    def build(self):
        model = keras.Sequential([
            layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_size, input_length=self.seq_length),
            layers.Conv1D(filters=50, kernel_size=5, strides=1, padding='valid'),
            layers.MaxPool1D(2, padding='valid'),
            layers.Flatten(),
            layers.Dense(5, activation='relu'),
            layers.Softmax(1)
        ])
        model.compile(optimizer=keras.optimizers.Adam(1e-3),
                      loss=keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy'])

        return model

