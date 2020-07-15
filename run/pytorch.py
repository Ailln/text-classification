import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from sklearn import metrics

from model.pytorch import cnn
from model.pytorch import rnn
from util import conf_utils
from util import data_utils
from util import batch_utils
from util import output_utils

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--run", default="train", help="run type")
parser.add_argument("-c", "--config", default="./config/pytorch.yaml", help="run config")
parser.add_argument("-m", "--model", default="cnn", help="model name")
parser.add_argument("-d", "--date", default="", help="date file name")
parser.add_argument("-t", "--text", default="", help="infer text")
args = parser.parse_args()

# 判断是否有GPU加速
use_gpu = torch.cuda.is_available()
print(f">> use gpu: {use_gpu}")


def train(config_path):
    config = conf_utils.init_train_config(config_path)

    batch_size = config["batch_size"]
    epoch_size = config["epoch_size"]
    model_class = config["model_class"]
    print(f"\n>> model class is {model_class}")

    train_input, train_target, val_input, val_target = data_utils.gen_train_data(config)
    if args.model == "cnn":
        model = cnn.Model(config)
    elif args.model == "rnn":
        model = rnn.Model(config)
    else:
        raise Exception(f"error model: {args.model}")

    if use_gpu:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])

    for epoch in range(epoch_size):
        epoch = epoch + 1
        train_batch_gen = batch_utils.make_batch(train_input, train_target, batch_size)
        val_batch_gen = batch_utils.make_batch(val_input, val_target, batch_size)

        train_acc = 0
        val_acc = 0
        for batch_num in range(len(train_input) // batch_size):
            train_input_batch, train_target_batch = train_batch_gen.__next__()

            if args.model == "cnn" and model_class == "conv2d":
                train_input_batch_v = Variable(torch.LongTensor(np.expand_dims(train_input_batch, 1)))
            else:
                train_input_batch_v = Variable(torch.LongTensor(train_input_batch))

            train_target_batch_v = Variable(torch.LongTensor(np.argmax(train_target_batch, 1)))

            if use_gpu:
                train_input_batch_v = train_input_batch_v.cuda()
                train_target_batch_v = train_target_batch_v.cuda()

            out = model(train_input_batch_v)
            loss = criterion(out, train_target_batch_v)
            _, pred = torch.max(out, 1)
            train_acc = (pred == train_target_batch_v).float().mean()

            # 后向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not batch_num % 10:
                with torch.no_grad():
                    val_input_batch, val_target_batch = val_batch_gen.__next__()
                    if args.model == "cnn" and model_class == "conv2d":
                        val_input_batch_v = Variable(torch.LongTensor(np.expand_dims(val_input_batch, 1)))
                    else:
                        val_input_batch_v = Variable(torch.LongTensor(val_input_batch))

                    val_target_batch_v = Variable(torch.LongTensor(np.argmax(val_target_batch, 1)))
                    if use_gpu:
                        val_input_batch_v = val_input_batch_v.cuda()
                        val_target_batch_v = val_target_batch_v.cuda()
                    val_out = model(val_input_batch_v)
                    _, val_pred = torch.max(val_out, 1)
                    val_acc = (val_pred == val_target_batch_v).float().mean()

                print(f">> e:{epoch:3} s:{batch_num:2} loss:{loss:5.4} train-acc:{train_acc:.4f} val-acc:{val_acc:.4f}")

        if epoch > 10 and train_acc >= 0.9 and val_acc >= 0.9:
            torch.save({
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "model": model.state_dict(),
                "train_acc": train_acc
            }, config["model_path"] + f"{args.model}.pt")


def test(config_path):
    config = conf_utils.init_test_config(config_path)

    batch_size = config["batch_size"]
    model_class = config["model_class"]
    if args.model == "cnn":
        print(f"\n>> model class is {model_class}")

    test_data, test_target = data_utils.gen_test_data(config)

    model = cnn.Model(config)
    if use_gpu:
        model = model.cuda()

    model.load_state_dict(torch.load(config["model_path"] + "cnn.pt", map_location=torch.device('cpu'))["model"])

    test_batch_gen = batch_utils.make_batch(test_data, test_target, batch_size)

    pred_list = []
    target_list = []
    for batch_num in range(len(test_data) // batch_size):
        test_input_batch, test_target_batch = test_batch_gen.__next__()

        if args.model == "cnn" and model_class == "conv2d":
            test_input_batch_v = Variable(torch.LongTensor(np.expand_dims(test_input_batch, 1)))
        else:
            test_input_batch_v = Variable(torch.LongTensor(test_input_batch))

        if use_gpu:
            test_input_batch_v = test_input_batch_v.cuda()

        # 向前传播
        out = model(test_input_batch_v)
        _, pred = torch.max(out, 1)
        if use_gpu:
            pred = pred.cpu().numpy()
        pred_list.extend(pred)
        target_list.extend([np.argmax(target) for target in test_target_batch])

    report = metrics.classification_report(target_list, pred_list)
    print(f"\n>> REPORT:\n{report}")
    output_utils.save_metrics(config, "report.txt", report)

    cm = metrics.confusion_matrix(target_list, pred_list)
    print(f"\n>> Confusion Matrix:\n{cm}")
    output_utils.save_metrics(config, "confusion_matrix.txt", str(cm))


def infer(text):
    config = conf_utils.init_test_config("20-07-15_05-08-14")
    model_class = config["model_class"]

    model = cnn.Model(config)
    if use_gpu:
        model = model.cuda()

    model.load_state_dict(torch.load(config["model_path"] + "cnn.pt", map_location=torch.device('cpu'))["model"])

    infer_input_batch, target_vocab = data_utils.gen_infer_data(config, text)
    if args.model == "cnn" and model_class == "conv2d":
        infer_input_v = Variable(torch.LongTensor(np.expand_dims(infer_input_batch, 1)))
    else:
        infer_input_v = Variable(torch.LongTensor(infer_input_batch))

    if use_gpu:
        infer_input_v = infer_input_v.cuda()

    # 向前传播
    out = model(infer_input_v)
    _, pred = torch.max(out, 1)

    target = target_vocab[pred[0]]
    print(f"\n\n>> {text}: {target}")
    return target


if __name__ == '__main__':
    if args.run == "train":
        train(args.config)
    elif args.run == "test":
        if args.date == "":
            raise ValueError("test need -d arg")
        else:
            test(args.date)
    elif args.run == "infer":
        if args.text == "":
            raise ValueError("infer need -t arg")
        else:
            infer(args.text)
    else:
        raise ValueError("The first parameter is wrong, only support train or test!")
