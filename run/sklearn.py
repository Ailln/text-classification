import argparse

import joblib
from sklearn import metrics

from util import data_utils
from util import conf_utils
from util import output_utils
from model.sklearn import svm
from model.sklearn import bayes

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--run", default="train", help="run type")
parser.add_argument("-c", "--config", default="./config/sklearn.yaml", help="run config")
parser.add_argument("-m", "--model", default="svm", help="model name")
parser.add_argument("-d", "--date", default="", help="date file name")
parser.add_argument("-t", "--text", default="", help="infer text")
args = parser.parse_args()


def train(config_path):
    config = conf_utils.init_train_config(config_path)

    train_input, train_target, validate_input, validate_target = data_utils.gen_train_data(config)

    if args.model == "bayes":
        model = bayes.Model()
        train_term_doc = model.get_tfidf(train_input, "train")
        validate_term_doc = model.get_tfidf(validate_input, "validate")
        m = model.multi_nb()
        m.fit(train_term_doc, train_target)
    elif args.model == "svm":
        model = svm.Model()
        train_term_doc = model.get_tfidf(train_input, "train")
        validate_term_doc = model.get_tfidf(validate_input, "validate")
        m = model.lin_svm()
        m.fit(train_term_doc, train_target)
    else:
        raise Exception(f"error model name: {args.model}")

    # save vocab
    vocab = model.vec.vocabulary_
    joblib.dump(vocab, config["model_path"] + "vocab.json")

    # save model
    joblib.dump(m, config["model_path"] + f"{args.model}.m")

    # validate
    validate_preds = m.predict(validate_term_doc)

    report = metrics.classification_report(validate_target, validate_preds)
    print(f"\n>> REPORT:\n{report}")

    cm = metrics.confusion_matrix(validate_target, validate_preds)
    print(f"\n>> Confusion Matrix:\n{cm}")


def test(time_str):
    config = conf_utils.init_test_config(time_str)

    test_input, test_target = data_utils.gen_test_data(config)

    vocab = joblib.load(config["model_path"] + "vocab.json")

    model = bayes.Model(vocab)
    test_term_doc = model.get_tfidf(test_input, "test")

    m = joblib.load(config["model_path"] + f"{args.model}.m")

    test_preds = m.predict(test_term_doc)

    report = metrics.classification_report(test_target, test_preds)
    print(f"\n>> REPORT:\n{report}")
    output_utils.save_metrics(config, "report.txt", report)

    cm = metrics.confusion_matrix(test_target, test_preds)
    print(f"\n>> Confusion Matrix:\n{cm}")
    output_utils.save_metrics(config, "confusion_matrix.txt", str(cm))


if __name__ == '__main__':
    if args.run == "train":
        train(args.config)
    elif args.run == "test":
        if args.date == "":
            raise ValueError("test need -d arg")
        else:
            test(args.date)
    else:
        raise ValueError("The first parameter is wrong, only support train or test!")
