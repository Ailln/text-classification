import sys

from sklearn.externals import joblib
from sklearn import metrics

from util import data_utils
from util import conf_utils
from util import output_utils
from model.sklearn import bayes


def train(config_path):
    config = conf_utils.init_train_config(config_path)

    train_input, train_target, validate_input, validate_target = data_utils.gen_train_data(config)

    model = bayes.Model()
    train_term_doc = model.get_tfidf(train_input, "train")
    validate_term_doc = model.get_tfidf(validate_input, "validate")
    nb = model.multi_nb()
    nb.fit(train_term_doc, train_target)

    # save vocab
    vocab = model.vec.vocabulary_
    joblib.dump(vocab, config["model_path"] + "vocab.json")

    # save model
    joblib.dump(nb, config["model_path"] + "svm.m")

    # validate
    validate_preds = nb.predict(validate_term_doc)

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

    nb = joblib.load(config["model_path"] + "svm.m")

    test_preds = nb.predict(test_term_doc)

    report = metrics.classification_report(test_target, test_preds)
    print(f"\n>> REPORT:\n{report}")
    output_utils.save_metrics(config, "report.txt", report)

    cm = metrics.confusion_matrix(test_target, test_preds)
    print(f"\n>> Confusion Matrix:\n{cm}")
    output_utils.save_metrics(config, "confusion_matrix.txt", str(cm))


if __name__ == '__main__':
    args = sys.argv
    if len(args) == 1:
        config_data_path = "./config/sklearn-bayes.yaml"
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
