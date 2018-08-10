import numpy as np
import tensorflow as tf

from util import data_utils
from model.tensorflow import cnn

if __name__ == '__main__':
    config_file_path = "./config/tensorflow-cnn.yaml"
    config = data_utils.get_config(config_file_path)
    train_input, train_target, test_input, test_target = data_utils.gen_train_data(config)

    print(">> build model...")
    model = cnn.Model(config)
    train_step, pred, loss = model.cnn()
    print(">> start tf...")
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        batch_size = 8
        epoch_size = 100
        for epoch in range(epoch_size):
            print(f">> epoch: {epoch}")
            for batch_num in range(len(train_input)//batch_size):

                train_input_batch = train_input[batch_size*batch_num:batch_size*(batch_num+1)]
                train_target_batch = train_target[batch_size*batch_num:batch_size*(batch_num+1)]

                sess.run([train_step], feed_dict={
                    model.input_holder: train_input_batch,
                    model.target_holder: train_target_batch
                })

                loss_result, pred_result = sess.run([loss, pred], feed_dict={
                    model.input_holder: train_input_batch,
                    model.target_holder: train_target_batch
                })
                if not batch_num % 10:
                    print(f">> step: {str(batch_num)} loss: {loss_result}")

                print(np.argmax(train_target_batch, 1))
                print(pred_result)
