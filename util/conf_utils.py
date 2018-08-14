from pathlib import Path
from datetime import datetime
import yaml

from util import output_utils


def read_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f_config:
        config_data = yaml.load(f_config.read())
    return config_data


def init_train_config(config_path):
    time_now = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    print(f">> start time: {time_now}\n")

    config = read_config(config_path)
    config["config_path"] = config_path
    config["time_now"] = time_now
    config["output_path_with_time"] = config["output_path"] + config["time_now"] + "/"
    config["log_train_path"] = config["output_path_with_time"]+"log/"
    config["model_path"] = config["output_path_with_time"]+"model/"

    output_utils.check_path(config["log_train_path"])
    output_utils.check_path(config["model_path"])
    output_utils.cp_config(config)
    output_utils.cp_data(config)
    return config


def init_test_config(train_time_str):
    time_now = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    print(f">> start time: {time_now}\n")

    test_config_path = list(Path(f"./output/{train_time_str}/config/").glob("*.yaml"))[0]
    config = read_config(test_config_path)

    config["test_path_with_time"] = config["output_path_with_time"] + "test/" + time_now + "/"

    output_utils.cp_model(config)
    return config
