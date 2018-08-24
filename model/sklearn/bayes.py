from sklearn import naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer


class Model(object):
    def __init__(self, vocab=None):
        self.vec = TfidfVectorizer(vocabulary=vocab)

    def get_tfidf(self, data, run_type):
        if run_type in ["train", "test"]:
            term_doc = self.vec.fit_transform(data)
        elif run_type in ["validate"]:
            term_doc = self.vec.transform(data)
        else:
            raise ValueError
        return term_doc

    @staticmethod
    def multi_nb():
        nb = naive_bayes.MultinomialNB()
        return nb
