from datetime import datetime

import numpy as np
import tensorflow as tf

from util import data_utils
from util import batch_utils
from util import output_utils
from model.tensorflow import cnn


def train():
    time_now = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    print(f">> start time: {time_now}\n")

    config_path = "./config/tensorflow-cnn.yaml"
    config = data_utils.get_config(config_path)
    config["config_path"] = config_path
    config["time_now"] = time_now
    config["output_path_with_time"] = config["output_path"] + config["time_now"] + "/"
    config["log_train_path"] = config["output_path_with_time"]+"log/train/"
    config["log_test_path"] = config["output_path_with_time"]+"log/test/"

    output_utils.check_path(config["log_train_path"])
    output_utils.check_path(config["log_test_path"])
    output_utils.cp_config(config)
    output_utils.cp_data(config)

    batch_size = config["batch_size"]
    epoch_size = config["epoch_size"]

    train_input, train_target, validate_input, validate_target = data_utils.gen_train_data(config)

    validate_input_batch = validate_input[:batch_size]
    validate_target_batch = validate_target[:batch_size]

    print(">> build model...")
    model = cnn.Model(config)
    train_step, pred, cost = model.cnn()
    print(">> start tf...")
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(config["log_train_path"], sess.graph)

        for epoch in range(epoch_size):
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


if __name__ == '__main__':
    train()
