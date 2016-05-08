# -*- coding: utf-8 -*-

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, recall_score, precision_score

import bald_latin


def round(labels):
    return np.round(labels).astype(np.int)


class IterativeNB(object):

    def __init__(self):
        # The clean corpus, labels, and scraped comments
        self.corpus = None
        self.labels = None
        self.scraped = None

        # The training and testing corpus and labels which will change
        # with each iteration
        self.corpus_train = None
        self.labels_train = None
        self.corpus_test = None
        self.labels_test = None

        # The tf-idf vectors for the training set, the test set, and scraped comments
        self.X_train = None
        self.X_test = None
        self.X_scraped = None

        self.vectorizer = self._build_vectorizer()
        self.clf = MultinomialNB()

    @staticmethod
    def _build_vectorizer():
        """
        Builds a TF-IDF vectorizer with croatian stop words.
        Uses bigrams. Counts words appearing 2+ times.
        """
        croatian_stop_words = set([u"a", u"ako", u"ali", u"bi", u"bih", u"bila", u"bili", u"bilo", u"bio", u"bismo", u"biste", u"biti", u"bumo", u"da", u"do", u"duž", u"ga", u"hoće", u"hoćemo", u"hoćete", u"hoćeš", u"hoću", u"i", u"iako", u"ih", u"ili", u"iz", u"ja", u"je", u"jedna", u"jedne", u"jedno", u"jer", u"jesam", u"jesi", u"jesmo", u"jest", u"jeste", u"jesu", u"jim", u"joj", u"još", u"ju", u"kada", u"kako", u"kao", u"koja", u"koje", u"koji", u"kojima", u"koju", u"kroz", u"li", u"me", u"mene", u"meni", u"mi", u"mimo", u"moj", u"moja", u"moje", u"mu", u"na", u"nad", u"nakon", u"nam", u"nama", u"nas", u"naš", u"naša", u"naše", u"našeg", u"ne", u"nego", u"neka", u"neki", u"nekog", u"neku", u"nema", u"netko", u"neće", u"nećemo", u"nećete", u"nećeš", u"neću", u"nešto", u"ni", u"nije", u"nikoga", u"nikoje", u"nikoju", u"nisam", u"nisi", u"nismo", u"niste", u"nisu", u"njega", u"njegov", u"njegova", u"njegovo", u"njemu", u"njezin", u"njezina", u"njezino", u"njih", u"njihov", u"njihova", u"njihovo", u"njim", u"njima", u"njoj", u"nju", u"no", u"o", u"od", u"odmah", u"on", u"ona", u"oni", u"ono", u"ova", u"pa", u"pak", u"po", u"pod", u"pored", u"prije", u"s", u"sa", u"sam", u"samo", u"se", u"sebe", u"sebi", u"si", u"smo", u"ste", u"su", u"sve", u"svi", u"svog", u"svoj", u"svoja", u"svoje", u"svom", u"ta", u"tada", u"taj", u"tako", u"te", u"tebe", u"tebi", u"ti", u"to", u"toj", u"tome", u"tu", u"tvoj", u"tvoja", u"tvoje", u"u", u"uz", u"vam", u"vama", u"vas", u"vaš", u"vaša", u"vaše", u"već", u"vi", u"vrlo", u"za", u"zar", u"će", u"ćemo", u"ćete", u"ćeš", u"ću", u"što"])

        # vectorize the text
        vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            ngram_range=(1, 2),  # unigrams and bigrams
            min_df=2,  # words must occur 2+ times to be counted
            norm='l2',
            smooth_idf=True,
            use_idf=True,
            stop_words=croatian_stop_words)

        return vectorizer

    def load_dataset(self, train_len=4000, max_scraped=None):
        """
        Load the corpus, labels and scraped comments into lists.
        Removes cyrillic and baldens corpus and scraped lists. 
        Rounds and converts the comments to ints.
        """
        # the number of samples for the training set
        self._train_len = train_len

        # Load the labeled dataset
        self.corpus = open('dataset/lns_stemmed.txt', 'r').readlines()
        self.labels = open('dataset/lns_labels.txt', 'r').readlines()

        # Load the unlabeled comments
        self.scraped = open('dataset/stemmed.txt', 'r').readlines()
        if max_scraped is not None and max_scraped > 0:
            self.scraped = self.scraped[:max_scraped]

        # remove cyrillic and balden text
        self.corpus, self.labels = bald_latin.remove_cyrillic_comments(self.corpus, self.labels)
        self.corpus = bald_latin.remove_serbian_accents(self.corpus)
        self.scraped, _ = bald_latin.remove_cyrillic_comments(self.scraped, range(len(self.scraped)))
        self.scraped = bald_latin.remove_serbian_accents(self.scraped)

        # labels as a numpy array
        self.labels = np.array([int(float(x)) for x in self.labels])

        # convert to tuples for immutability
        self.corpus = tuple(self.corpus)
        self.labels = self.labels
        self.scraped = tuple(self.scraped)

    def _build_test_set(self):
        """
        The training set takes the first self._train_len comments and labels.
        The test set receives every comment after train_len.
        """
        self.corpus_test = self.corpus[self._train_len:]
        self.labels_test = self.labels[self._train_len:]

    def _build_next_training_set(self, scraped_indices=None, scraped_labels=None):
        """
        Creates the next training set from the first self._train_len comments,
        and a subset of scraped comments.
        The indices of scraped comments are passed in, along with their current labels.
        The labels of scraped comments may change during further classifications.
        """
        self.corpus_train = list(self.corpus[:self._train_len])
        self.labels_train = list(self.labels[:self._train_len])

        if scraped_indices is not None and len(scraped_indices) > 0:
            assert scraped_indices.shape[0] == len(scraped_labels)
            self.corpus_train += [self.scraped[x] for x in scraped_indices]
            self.labels_train = self.labels_train + list(scraped_labels)

        assert len(self.corpus_train) == len(self.labels_train)

    def vectorize_all_sets(self, prnt=False):
        """
        Fit the vectorizer on the training set, get the TF-IDF vectors
        for the training set, the test set, and the unlabeled comments.
        """
        if prnt:
            print("\tVectorizing the training set")
        #self.X_train = self.vectorizer.fit_transform(self.corpus_train).todense()
        self.X_train = self.vectorizer.fit_transform(self.corpus_train)
        if prnt:
            print("\tVectorizing the test set")
        self.X_test = self.vectorizer.transform(self.corpus_test)
        if prnt:
            print("\tVectorizing the scraped set")
        self.X_scraped = self.vectorizer.transform(self.scraped)

    def classify(self):
        """
        Create a Naive Bayes classifier and fit it on the X_train, labels_train pair.
        Run the classifier on X_train, X_test, and X_scraped.
        """
        print self.X_train.shape[0], len(self.labels_train)
        assert self.X_train.shape[0] == len(self.labels_train)
        # fit the classifier
        self.clf.fit(self.X_train, self.labels_train)

        y_train = self.clf.predict_proba(self.X_train)[:, 1]
        y_test = self.clf.predict_proba(self.X_test)[:, 1]
        y_scraped = self.clf.predict_proba(self.X_scraped)[:, 1]

        return y_train, y_test, y_scraped

    def prune_comments(self, y_pred, threshold=0.9, prnt=False):
        """
        Returns the indices of the comments with the highest certainty classifications.
        """
        indices = np.argwhere(np.logical_or(y_pred >= threshold, y_pred <= 1 - threshold))[:, 0]
        labels = round(y_pred[indices])
        if prnt: 
            print("\t{} comments added to the training set".format(len(indices)))

        return indices, labels

    def test_clf(self, y_true, y_pred, prnt=True):
        y_pred = round(np.copy(y_pred))
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        
        if prnt:
            print("\tAccuracy on the test set:  {}".format(accuracy))
            print("\tPrecision on the test set: {}".format(precision))
            print("\tRecall on the test set:    {}".format(recall))

        return accuracy, precision, recall

    def train_classifier(self, iterations=100, prnt=True):
        """
        Iteratively retrains the classifier on the original training set and newly labeled comments from the scraped dataset.

        1. Set up the data, create the test set
        2. Create a new training set from the original training set and the most certain comments from the scraped dataset
        3. Vectorize the data, train the classifier
        4. Run the classifier on the scraped data
        5. Go to 2.
        """
        if prnt: print("Loading the dataset")
        self.load_dataset(max_scraped=100000)
        self._build_test_set()

        # for the first iteration we don't use any of the unlabeled comments
        best_indices = None
        y_scraped = None

        # run the classification for n iterations
        for iter in range(iterations): 
            if prnt: 
                print("Running iteration #{}".format(iter))
                
            self._build_next_training_set(best_indices, y_scraped)
            print "\tLength of the training set: {}".format(len(self.corpus_train))

            # fit the vectorizer on the training set, transform the test set and scraped
            self.vectorize_all_sets(prnt=False)

            # fit classifier on the (X_train, y_train), classify all three sets
            if prnt:
                print("\tClassifying three sets")
            y_train, y_test, y_scraped = self.classify()

            self.test_clf(self.labels_test, y_test)

            best_indices, y_scraped = self.prune_comments(y_scraped, prnt=prnt)
            y_scraped = round(y_scraped)
