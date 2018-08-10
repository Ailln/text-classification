import numpy as np
import tensorflow as tf

from util import data_utils
from model.tensorflow import cnn

if __name__ == '__main__':
    config_file_path = "./config/tensorflow-cnn.yaml"
    config = data_utils.get_config(config_file_path)
    batch_size = config["batch_size"]
    epoch_size = config["epoch_size"]
    train_input, train_target, test_input, test_target = data_utils.gen_train_data(config)

    test_input_batch = test_input[:batch_size]
    test_target_batch = test_target[:batch_size]

    print(">> build model...")
    model = cnn.Model(config)
    train_step, pred, loss = model.cnn()
    print(">> start tf...")
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(epoch_size):
            # ToDo shuffle
            for batch_num in range(len(train_input)//batch_size):
                train_input_batch = train_input[batch_size*batch_num:batch_size*(batch_num+1)]
                train_target_batch = train_target[batch_size*batch_num:batch_size*(batch_num+1)]

                _, loss_result = sess.run([train_step, loss], feed_dict={
                    model.input_holder: train_input_batch,
                    model.target_holder: train_target_batch
                })

                if not batch_num % 5:
                    pred_result = sess.run([pred], feed_dict={
                        model.input_holder: test_input_batch,
                        model.target_holder: test_target_batch
                    })
                    input_test_arr = np.argmax(test_target_batch, 1)
                    target_test_arr = np.array(pred_result)
                    acc = np.sum(input_test_arr == target_test_arr)/len(input_test_arr)
                    print(f">> epoch: {epoch} step: {batch_num:2} loss: {loss_result:.4} acc: {acc:.4}")
