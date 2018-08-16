import torch.nn as nn


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_length = config["seq_length"]
        self.embedding_size = config["embedding_size"]
        self.vocab_size = config["vocab_size"]
        self.num_classes = config["num_classes"]
        self.filter_size = config["filter_size"]
        self.kernel_size = config["kernel_size"]
        self.opt_name = config["opt_name"]
        self.lr = config["learning_rate"]

        self.conv = nn.Sequential(
            nn.Embedding(self.vocab_size, self.embedding_size),
            nn.Conv1d(1, self.filter_size, self.kernel_size),
            nn.MaxPool1d(self.seq_length)
        )

        self.fc = nn.Sequential(
            nn.Linear(self.filter_size, self.num_classes)
        )

    def forward(self, input_data):
        out = self.conv(input_data)
        out = self.fc(out)
        return out
