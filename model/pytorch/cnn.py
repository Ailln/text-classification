import torch.nn as nn


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_length = config["seq_length"]
        self.batch_size = config["batch_size"]
        self.embedding_size = config["embedding_size"]
        self.vocab_size = config["vocab_size"]
        self.num_classes = config["num_classes"]
        self.filter_size = config["filter_size"]
        self.kernel_size = config["kernel_size"]
        self.opt_name = config["opt_name"]
        self.lr = config["learning_rate"]
        self.dropout_keep_prob = config["dropout_keep_prob"]

        self.embed = nn.Embedding(self.vocab_size, self.embedding_size)

        self.conv2d = nn.Conv2d(1, self.filter_size, (self.kernel_size, self.embedding_size), padding=(2, 0))
        self.max_pool2d = nn.MaxPool2d((1, self.seq_length))

        self.conv1d_and_max_pool = nn.Sequential(
            nn.Conv1d(self.embedding_size, self.filter_size, self.kernel_size, padding=1),
            nn.MaxPool1d(self.seq_length)
        )
        self.fc_and_softmax = nn.Sequential(
            nn.Linear(self.filter_size, self.num_classes),
            nn.Softmax(1)
        )

    def forward(self, input_data, model_class):
        out = self.embed(input_data)
        if model_class == "conv2d":
            out = self.conv2d(out).squeeze()
            out = self.max_pool2d(out).squeeze()
        elif model_class == "conv1d":
            out = out.permute(0, 2, 1)
            out = self.conv1d_and_max_pool(out).squeeze()
        else:
            raise ValueError("model class is wrong!")

        out = self.fc_and_softmax(out)
        return out
