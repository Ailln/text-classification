from pathlib import Path

import yaml
import numpy as np


def get_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f_vocab:
        vocab_list = []
        for line in f_vocab:
            vocab_list.append(line.strip())
    return vocab_list


def read_data(input_data_path):
    path = Path(input_data_path)
    input_data_list = []
    target_data_list = []
    for input_path in path.glob("*.txt"):
        target_data = input_path.name.split("-")[0]
        with open(str(input_path), "r", encoding="utf-8") as f_file:
            input_data = ""
            for line in f_file:
                line = line.strip().replace(" ", "")
                input_data += line
        input_data_list.append(input_data)
        target_data_list.append(target_data)
    return input_data_list, target_data_list


def word2id(input_vocab, input_word_list, seq_length):
    all_input_id_list = np.zeros([len(input_word_list), seq_length])
    for i_input, word_item in enumerate(input_word_list):
        word_item_list = [word for word in word_item]
        word_item_len = len(word_item_list)

        if word_item_len >= seq_length:
            word_item_list = word_item_list[:seq_length]
        else:
            word_item_list = word_item_list + ["PAD"] * (seq_length-word_item_len)

        input_id_list = []
        for word in word_item_list:
            if word in input_vocab:
                word_id = input_vocab.index(word)
            else:
                word_id = input_vocab.index("UNK")
            input_id_list.append(word_id)
        all_input_id_list[i_input] = np.array(input_id_list)

        read_percent = i_input/(len(input_word_list)/100)
        if not i_input % 10:
            print(f">> convert: {read_percent:3.1f}%", end="\r")

    return all_input_id_list


def target2id(target_vocab, target_data):
    target_id_list = np.zeros([len(target_data), len(target_vocab)])
    for i_target, target_item in enumerate(target_data):
        if target_item in target_vocab:
            target_id = target_vocab.index(target_item)
            target_id_list[i_target][target_id] = 1
        else:
            raise ValueError(target_item)
    return target_id_list


def gen_train_data(config):
    print("\n>> start read train data...")
    train_input, train_target = read_data(config["train_data_path"])
    print(">> start read validate data...")
    validate_input, validate_target = read_data(config["validate_data_path"])

    if config["frame_class"] == "sklearn":
        return train_input, train_target,  validate_input, validate_target
    else:
        print("\n>> start read input vocab...")
        input_vocab = get_vocab(config["input_vocab_path"])
        print(">> start read target vocab...")
        target_vocab = get_vocab(config["target_vocab_path"])

        print("\n>> word to id: train...")
        train_input_list = word2id(input_vocab, train_input, config["seq_length"])
        print(">> word to id: validate...")
        validate_input_list = word2id(input_vocab, validate_input, config["seq_length"])

        print("\n>> target to id: train...")
        train_target_list = target2id(target_vocab, train_target)
        print(">> target to id: validate...")
        validate_target_list = target2id(target_vocab, validate_target)

        return train_input_list, train_target_list, validate_input_list, validate_target_list


def gen_test_data(config):
    print(">> start read input vocab...")
    input_vocab = get_vocab(config["input_vocab_path"])
    print(">> start read target vocab...")
    target_vocab = get_vocab(config["target_vocab_path"])

    print("\n>> start read test data...")
    test_input, test_target = read_data(config["test_data_path"])

    if config["frame_class"] == "sklearn":
        return test_input, test_target
    else:
        print("\n>> word to id: test...")
        test_input_list = word2id(input_vocab, test_input, config["seq_length"])
        print("\n>> target to id: test...")
        test_target_list = target2id(target_vocab, test_target)

        return test_input_list, test_target_list
