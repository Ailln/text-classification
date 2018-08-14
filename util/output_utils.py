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
    shutil.copytree(config["data_path"], target_path)


def cp_model(config):
    target_path = config["test_path_with_time"] + "model/"
    shutil.copytree(config["model_path"], target_path)


def save_report(config, report_data):
    save_path = config["test_path_with_time"] + "report.txt"
    with open(save_path, "w", encoding="utf-8") as f_save:
        f_save.write(report_data)
