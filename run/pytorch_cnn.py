import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from model.pytorch import cnn
from util import conf_utils
from util import data_utils
from util import batch_utils


def train(config_path):
    config = conf_utils.init_train_config(config_path)

    batch_size = config["batch_size"]
    epoch_size = config["epoch_size"]

    train_input, train_target, validate_input, validate_target = data_utils.gen_train_data(config)
    model = cnn.Model(config)
    use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
    if use_gpu:
        model = model.cuda()

    # 定义loss和optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])

    for epoch in range(epoch_size):
        epoch = epoch + 1
        batch_gen = batch_utils.make_batch(train_input, train_target, batch_size)

        for batch_num in range(len(train_input) // batch_size):
            train_input_batch, train_target_batch = batch_gen.__next__()
            train_input_batch_v = Variable(train_input_batch)
            train_target_batch_v = Variable(train_target_batch)
            if use_gpu:
                train_input_batch_v = train_input_batch_v.cuda()
                train_target_batch_v = train_target_batch_v.cuda()

            # 向前传播
            out = model(train_input_batch_v)
            loss = criterion(out, train_target_batch_v)
            _, pred = torch.max(out, 1)
            accuracy = (pred == train_target_batch_v).float().mean()

            # 向后传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not batch_num % 5:
                print(f">> e:{epoch:3} s:{batch_num:2} loss:{loss:5.4} acc: {accuracy:3f}")


def test(args2):
    pass


if __name__ == '__main__':
    args = sys.argv
    if len(args) == 1:
        config_data_path = "./config/pytorch-cnn.yaml"
        train(config_data_path)
    elif len(args) == 3:
        if args[1] == "train":
            train(args[2])
        elif args[1] == "test":
            test(args[2])
        else:
            raise ValueError("The first parameter is wrong, only support train or test!")
    else:
        raise ValueError("Incorrent parameter length!")
