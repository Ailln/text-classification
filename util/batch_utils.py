import numpy as np


def shuffle_batch(input_data, target_data):
    datas = []
    for i_data, t_data in zip(input_data, target_data):
        datas.append([i_data, t_data])

    np.random.shuffle(datas)

    input_data_shuffle = []
    target_data_shuffle = []
    for data in datas:
        input_data_shuffle.append(data[0])
        target_data_shuffle.append(data[1])

    return input_data_shuffle, target_data_shuffle


def make_batch(input_data, target_data, batch_size, is_shuffle=True):
    if is_shuffle:
        input_data, target_data = shuffle_batch(input_data, target_data)

    for step in range(len(input_data) // batch_size):
        s = batch_size * step
        e = batch_size * (step + 1)

        input_data_batch = input_data[s:e]
        target_data_batch = target_data[s:e]

        yield input_data_batch, target_data_batch
