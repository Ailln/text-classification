import random
from pathlib import Path

import yaml
import numpy as np


def get_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f_config:
        config_data = yaml.load(f_config.read())
    return config_data


def read_data(input_data_path):
    path = Path(input_data_path)
    input_data_list = []
    for input_path in path.glob("*.txt"):
        target_data = input_path.name.split("-")[0]
        with open(input_path, "r", encoding="utf-8") as f_file:
            input_data = ""
            for line in f_file:
                line = line.strip().replace(" ", "")
                input_data += line
        input_data_list.append([input_data, target_data])
    return input_data_list


def split_train_and_test(input_and_target_list, test_size=0.3):
    if isinstance(input_and_target_list, list):
        test_num = int(len(input_and_target_list)*test_size)
        train_data = input_and_target_list[:-test_num]
        test_data = input_and_target_list[-test_num:]
        random.shuffle(train_data)
        random.shuffle(test_data)

        train_input_data = []
        train_target_data = []
        for train_i, train_t in train_data:
            train_input_data.append(train_i)
            train_target_data.append(train_t)

        test_input_data = []
        test_target_data = []
        for test_i, test_t in test_data:
            test_input_data.append(test_i)
            test_target_data.append(test_t)
    else:
        raise TypeError

    return train_input_data, train_target_data, test_input_data, test_target_data


def get_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f_vocab:
        vocab_list = []
        for line in f_vocab:
            vocab_list.append(line.strip())
    return vocab_list


def word2id(input_vocab, input_word_list, max_len):
    all_input_id_list = np.zeros([len(input_word_list), max_len, 1])
    max_len_input = 0
    for i_input, word_item in enumerate(input_word_list):
        word_item_list = [word for word in word_item]
        word_item_len = len(word_item_list)

        if word_item_len > max_len_input:
            max_len_input = word_item_len

        if word_item_len >= max_len:
            word_item_list = word_item_list[:max_len]
        else:
            word_item_list = word_item_list + ["PAD"] * (max_len-word_item_len)

        input_id_list = []
        for word in word_item_list:
            if word in input_vocab:
                word_id = input_vocab.index(word)
            else:
                word_id = input_vocab.index("UNK")
            input_id_list.append([word_id])
        all_input_id_list[i_input] = np.array(input_id_list)

        read_percent = i_input/(len(input_word_list)/100)
        if not i_input % 10:
            print(f">> read data percent: {read_percent:4.2f}", end="\r")

    print(f">> max len: {max_len_input}")
    return all_input_id_list


def target2id(target_vacab, target_data):
    target_id_list = []
    for target_item in target_data:
        if target_item in target_vacab:
            target_id = target_vacab.index(target_item)
            target_one_hot = [0] * len(target_vacab)
            target_one_hot[target_id] = 1
            target_id_list.append(target_one_hot)
        else:
            raise ValueError(target_item)
    return target_id_list


def gen_train_data(config):
    print(">> start read input vocab...")
    input_vocab = get_vocab(config["input_vocab_path"])
    print(f">> input vacab size: {len(input_vocab)}")
    print(">> start read target vocab...")
    target_vocab = get_vocab(config["target_vocab_path"])
    print(f">> target vacab size: {len(target_vocab)}")
    all_input_data = read_data(config["data_path"])

    train_input, train_target, test_input, test_target = split_train_and_test(all_input_data)

    train_input_list = word2id(input_vocab, train_input, config["input_max_len"])
    test_input_list = word2id(input_vocab, test_input, config["input_max_len"])

    train_target_list = target2id(target_vocab, train_target)
    test_target_list = target2id(target_vocab, test_target)

    return train_input_list, train_target_list, test_input_list, test_target_list


if __name__ == '__main__':
    config_file_path = "../config/tensorflow-cnn.yaml"
    config_dict = get_config(config_file_path)
    gen_train_data(config_dict)
