import torch
import torch.nn as nn
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_length = config["seq_length"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
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
        self.rnn = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.fc_and_softmax = nn.Sequential(
            nn.Linear(self.seq_length * self.hidden_size, self.num_classes),
            nn.Softmax(1)
        )

    def forward(self, input_data):
        out = self.embed(input_data)
        out = out.permute(1, 0, 2)
        hidden = Variable(torch.zeros(self.num_layers, self.seq_length, self.hidden_size))
        if USE_CUDA:
            hidden = hidden.cuda()
        context = Variable(torch.zeros(self.num_layers, self.seq_length, self.hidden_size))
        if USE_CUDA:
            context = context.cuda()

        out, _ = self.rnn(out, (hidden, context))  # seq, batch, hidden
        out = out.permute(1, 0, 2)

        out = out.reshape(self.batch_size, self.seq_length * self.hidden_size)
        out = self.fc_and_softmax(out)
        return out
