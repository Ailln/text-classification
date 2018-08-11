import os
import shutil

import yaml


def check_path(path_name):
    if not os.path.exists(path_name):
        os.makedirs(path_name)
        print(f">> makedir {path_name}")


def cp_config(config):
    target_path = config["output_path_with_time"] + "config/"
    check_path(target_path)

    save_path = target_path + config["config_path"].split("/")[-1]
    save_dump = yaml.dump(config, default_flow_style=False, allow_unicode=True, encoding=None)
    with open(save_path, "w", encoding="utf-8") as f_save:
        f_save.write(save_dump)


def cp_data(config):
    target_path = config["output_path_with_time"] + "data/"
    check_path(target_path)
    input_vocab_path = config["data_path"] + "input_vocab.txt"
    shutil.copy(input_vocab_path, target_path)
    target_vocab_path = config["data_path"] + "target_vocab.txt"
    shutil.copy(target_vocab_path, target_path)

