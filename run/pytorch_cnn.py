import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from sklearn import metrics


from model.pytorch import cnn
from util import conf_utils
from util import data_utils
from util import batch_utils
from util import output_utils


def train(config_path):
    config = conf_utils.init_train_config(config_path)

    batch_size = config["batch_size"]
    epoch_size = config["epoch_size"]
    model_class = config["model_class"]
    print(f"\n>> model class is {model_class}")

    train_input, train_target, validate_input, validate_target = data_utils.gen_train_data(config)

    model = cnn.Model(config)

    # 判断是否有GPU加速
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])

    for epoch in range(epoch_size):
        epoch = epoch + 1
        train_batch_gen = batch_utils.make_batch(train_input, train_target, batch_size)

        train_acc = 0
        for batch_num in range(len(train_input) // batch_size):
            train_input_batch, train_target_batch = train_batch_gen.__next__()

            if model_class == "conv2d":
                train_input_batch_v = Variable(torch.LongTensor(np.expand_dims(train_input_batch, 1)))
            elif model_class == "conv1d":
                train_input_batch_v = Variable(torch.LongTensor(train_input_batch))
            else:
                raise ValueError("model class is wrong!")
            train_target_batch_v = Variable(torch.LongTensor(np.argmax(train_target_batch, 1)))

            if use_gpu:
                train_input_batch_v = train_input_batch_v.cuda()
                train_target_batch_v = train_target_batch_v.cuda()

            # 向前传播
            out = model(train_input_batch_v, model_class)
            loss = criterion(out, train_target_batch_v)
            _, pred = torch.max(out, 1)
            train_acc = (pred == train_target_batch_v).float().mean()

            # 向后传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not batch_num % 5:
                print(f">> e:{epoch:3} s:{batch_num:2} loss:{loss:5.4} acc: {train_acc:3f}")

        if epoch > 10 and train_acc >= 0.9:
            torch.save({
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "model": model.state_dict(),
                "train_acc": train_acc
            }, config["model_path"]+"cnn.pt")


def test(config_path):
    config = conf_utils.init_test_config(config_path)

    batch_size = config["batch_size"]
    test_data, test_target = data_utils.gen_test_data(config)

    model = cnn.Model(config)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()

    model.load_state_dict(torch.load(config["model_path"]+"cnn.pt", map_location=torch.device('cpu'))["model"])

    test_batch_gen = batch_utils.make_batch(test_data, test_target, batch_size)

    pred_list = []
    target_list = []
    for batch_num in range(len(test_data) // batch_size):
        test_input_batch, test_target_batch = test_batch_gen.__next__()

        test_input_batch_v = Variable(torch.LongTensor(np.expand_dims(test_input_batch, 1)))

        if use_gpu:
            test_input_batch_v = test_input_batch_v.cuda()

        # 向前传播
        out = model(test_input_batch_v)
        _, pred = torch.max(out, 1)
        pred_list.extend(pred)
        target_list.extend(test_target_batch)

    report = metrics.classification_report(target_list, pred_list)
    print(f"\n>> REPORT:\n{report}")
    output_utils.save_metrics(config, "report.txt", report)

    cm = metrics.confusion_matrix(target_list, pred_list)
    print(f"\n>> Confusion Matrix:\n{cm}")
    output_utils.save_metrics(config, "confusion_matrix.txt", str(cm))


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
