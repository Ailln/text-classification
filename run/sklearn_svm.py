import sys

from sklearn.externals import joblib
from sklearn import metrics

from util import data_utils
from util import conf_utils
from util import output_utils
from model.sklearn import svm


def train(config_path):
    config = conf_utils.init_train_config(config_path)

    train_input, train_target, validate_input, validate_target = data_utils.gen_train_data(config)

    model = svm.Model()
    train_term_doc = model.get_tfidf(train_input, "train")
    validate_term_doc = model.get_tfidf(validate_input, "validate")
    lin_svm = model.lin_svm()
    lin_svm.fit(train_term_doc, train_target)

    # save vocab
    vocab = model.vec.vocabulary_
    joblib.dump(vocab, config["model_path"] + "vocab.json")

    # save model
    joblib.dump(lin_svm, config["model_path"] + "svm.m")

    # validate
    validate_preds = lin_svm.predict(validate_term_doc)

    report = metrics.classification_report(validate_target, validate_preds)
    print(f"\n>> REPORT:\n{report}")

    cm = metrics.confusion_matrix(validate_target, validate_preds)
    print(f"\n>> Confusion Matrix:\n{cm}")


def test(time_str):
    config = conf_utils.init_test_config(time_str)

    test_input, test_target = data_utils.gen_test_data(config)

    vocab = joblib.load(config["model_path"] + "vocab.json")

    model = svm.Model(vocab)
    test_term_doc = model.get_tfidf(test_input, "test")

    lin_svm = joblib.load(config["model_path"] + "svm.m")

    test_preds = lin_svm.predict(test_term_doc)

    report = metrics.classification_report(test_target, test_preds)
    print(f"\n>> REPORT:\n{report}")
    output_utils.save_metrics(config, "report.txt", report)

    cm = metrics.confusion_matrix(test_target, test_preds)
    print(f"\n>> Confusion Matrix:\n{cm}")
    output_utils.save_metrics(config, "confusion_matrix.txt", str(cm))


if __name__ == '__main__':
    args = sys.argv
    if len(args) == 1:
        config_data_path = "./config/sklearn-svm.yaml"
        train(config_data_path)
    elif len(args) == 3:
        if args[1] == "train":
            train(args[2])
        elif args[1] == "test":
            test(args[2])
        else:
            raise ValueError("The first parameter is wrong, only support train or test!")
    else:
        raise ValueError("Incorrent parameter length!")
