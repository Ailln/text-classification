import sys

import numpy as np
import tensorflow as tf
from sklearn import metrics

from util import data_utils
from util import batch_utils
from util import conf_utils
from util import output_utils
from model.tensorflow import cnn


def train(config_path):
    config = conf_utils.init_train_config(config_path)
    batch_size = config["batch_size"]
    epoch_size = config["epoch_size"]
    num_save_epoch = config["num_save_epoch"]

    train_input, train_target, validate_input, validate_target = data_utils.gen_train_data(config)

    validate_input_batch = validate_input[:batch_size]
    validate_target_batch = validate_target[:batch_size]

    print(">> build model...")
    model = cnn.Model(config)
    train_step, pred, cost = model.cnn()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(config["log_train_path"], sess.graph)

        max_val_acc = 0
        max_key = 0
        for epoch in range(epoch_size):
            epoch = epoch + 1
            batch_gen = batch_utils.make_batch(train_input, train_target, batch_size)
            for batch_num in range(len(train_input)//batch_size):
                train_input_batch, train_target_batch = batch_gen.__next__()

                _, loss, train_pred, summary = sess.run([train_step, cost, pred, merged], feed_dict={
                    model.input_holder: train_input_batch,
                    model.target_holder: train_target_batch
                })
                train_writer.add_summary(summary)

                if not batch_num % 5:
                    input_train_arr = np.argmax(train_target_batch, 1)
                    target_train_arr = np.array(train_pred)
                    acc_train = np.sum(input_train_arr == target_train_arr)*100/len(input_train_arr)

                    validate_pred = sess.run([pred], feed_dict={
                        model.input_holder: validate_input_batch,
                        model.target_holder: validate_target_batch
                    })
                    input_validate_arr = np.argmax(validate_target_batch, 1)
                    target_validate_arr = np.array(validate_pred)
                    acc_val = np.sum(input_validate_arr == target_validate_arr)*100/len(input_validate_arr)
                    print(f">> e:{epoch:3} s:{batch_num:2} loss:{loss:5.4} acc_t: {acc_train:3f} acc_v: {acc_val:3f}")

                    if acc_val > max_val_acc:
                        max_val_acc = acc_val
                        max_key = 0
                    else:
                        max_key += 1

            if not epoch % num_save_epoch:
                saver.save(sess, config["model_path"]+"model", global_step=epoch)
                print(">> save model...")

            # 1000 batch val acc 没有增长，提前停止
            if max_key > 200:
                print(">> No optimization for a long time, auto stopping...")
                break

        time_str = config["time_now"]
        print(f">> use this command for test:\npython -m run.tensorflow_cnn test {time_str} ")


def test(time_str):
    config = conf_utils.init_test_config(time_str)
    batch_size = config["batch_size"]

    test_input, test_target = data_utils.gen_test_data(config)
    target_vocab = data_utils.get_vocab(config["target_vocab_path"])

    print(">> build model...")
    model = cnn.Model(config)
    _, pred, _ = model.cnn()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        lastest_checkpoint_name = tf.train.latest_checkpoint(config["model_path"])
        print(f">> last checkpoint: {lastest_checkpoint_name}")
        saver.restore(sess, lastest_checkpoint_name)

        batch_gen = batch_utils.make_batch(test_input, test_target, batch_size, False)
        input_target_list = []
        pred_target_list = []
        for batch_num in range(len(test_input)//batch_size):
            test_input_batch, test_target_batch = batch_gen.__next__()

            pred_target_arr = sess.run(pred, feed_dict={
                model.input_holder: test_input_batch,
                model.target_holder: test_target_batch
            })

            input_target_arr = np.argmax(test_target_batch, 1)
            input_target_list.extend(input_target_arr.tolist())
            pred_target_list.extend(pred_target_arr.tolist())

        input_target_list = [target_vocab[i_data] for i_data in input_target_list]
        pred_target_list = [target_vocab[p_data] for p_data in pred_target_list]
        report = metrics.classification_report(input_target_list, pred_target_list)
        print(f"\n>> REPORT:\n{report}")
        output_utils.save_metrics(config, "report.txt", report)

        cm = metrics.confusion_matrix(input_target_list, pred_target_list)
        print(f"\n>> Confusion Matrix:\n{cm}")
        output_utils.save_metrics(config, "confusion_matrix.txt", str(cm))


def get_server_sess(time_str):
    config = conf_utils.init_test_config(time_str)
    input_vocab = data_utils.get_vocab(config["input_vocab_path"])
    target_vocab = data_utils.get_vocab(config["target_vocab_path"])

    print(">> build model...")
    model = cnn.Model(config)
    _, pred, _ = model.cnn()

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    lastest_checkpoint_name = tf.train.latest_checkpoint(config["model_path"])
    print(f">> last checkpoint: {lastest_checkpoint_name}")
    saver.restore(sess, lastest_checkpoint_name)
    return sess, pred, target_vocab, input_vocab, model


if __name__ == '__main__':
    args = sys.argv
    if len(args) == 1:
        config_data_path = "./config/tensorflow-cnn.yaml"
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
