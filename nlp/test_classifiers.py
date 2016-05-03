# -*- coding: utf-8 -*-
import os

import numpy as np
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import bald_latin

# Load the dataset
originals = open('dataset/lns_comments.txt', 'r').readlines()
corpus = open('dataset/lns_stemmed.txt', 'r').readlines()
labels = open('dataset/lns_labels.txt', 'r').readlines()

# remove cyrillic and balden text
corpus, labels = bald_latin.remove_cyrillic_comments(corpus, labels)
originals, _ = bald_latin.remove_cyrillic_comments(originals, range(len(originals)))
corpus = bald_latin.remove_serbian_accents(corpus)

# labels as a numpy array
labels = np.array([int(float(x)) for x in labels])


# vectorize the text
croatian_stop_words = set([u"a",u"ako",u"ali",u"bi",u"bih",u"bila",u"bili",u"bilo",u"bio",u"bismo",u"biste",u"biti",u"bumo",u"da",u"do",u"duž",u"ga",u"hoće",u"hoćemo",u"hoćete",u"hoćeš",u"hoću",u"i",u"iako",u"ih",u"ili",u"iz",u"ja",u"je",u"jedna",u"jedne",u"jedno",u"jer",u"jesam",u"jesi",u"jesmo",u"jest",u"jeste",u"jesu",u"jim",u"joj",u"još",u"ju",u"kada",u"kako",u"kao",u"koja",u"koje",u"koji",u"kojima",u"koju",u"kroz",u"li",u"me",u"mene",u"meni",u"mi",u"mimo",u"moj",u"moja",u"moje",u"mu",u"na",u"nad",u"nakon",u"nam",u"nama",u"nas",u"naš",u"naša",u"naše",u"našeg",u"ne",u"nego",u"neka",u"neki",u"nekog",u"neku",u"nema",u"netko",u"neće",u"nećemo",u"nećete",u"nećeš",u"neću",u"nešto",u"ni",u"nije",u"nikoga",u"nikoje",u"nikoju",u"nisam",u"nisi",u"nismo",u"niste",u"nisu",u"njega",u"njegov",u"njegova",u"njegovo",u"njemu",u"njezin",u"njezina",u"njezino",u"njih",u"njihov",u"njihova",u"njihovo",u"njim",u"njima",u"njoj",u"nju",u"no",u"o",u"od",u"odmah",u"on",u"ona",u"oni",u"ono",u"ova",u"pa",u"pak",u"po",u"pod",u"pored",u"prije",u"s",u"sa",u"sam",u"samo",u"se",u"sebe",u"sebi",u"si",u"smo",u"ste",u"su",u"sve",u"svi",u"svog",u"svoj",u"svoja",u"svoje",u"svom",u"ta",u"tada",u"taj",u"tako",u"te",u"tebe",u"tebi",u"ti",u"to",u"toj",u"tome",u"tu",u"tvoj",u"tvoja",u"tvoje",u"u",u"uz",u"vam",u"vama",u"vas",u"vaš",u"vaša",u"vaše",u"već",u"vi",u"vrlo",u"za",u"zar",u"će",u"ćemo",u"ćete",u"ćeš",u"ću",u"što"])

vectorizer = TfidfVectorizer(
    strip_accents="unicode",
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    norm='l2',
    smooth_idf=True,
    use_idf=True,
    stop_words=croatian_stop_words)

X = vectorizer.fit_transform(corpus).todense()
y = labels

# X and labels are the inputs to the classifiers
# Here you can change the crossvalidation or something


def test_high_prob_predictions(proba, y, min_cert, prnt, cls_name):
    """
    Tests only the probabilities above min_cert.
    """
    assert len(proba) == len(y)
    indices = np.logical_or(proba >= min_cert, proba <= 1 - min_cert)

    error = np.mean(np.round(proba[indices]) == y[indices])
    count = float(np.sum(indices)) / len(indices) * 100

    if prnt:
        print "%s --- accuracy: %.2f, classified: %d, perc classified: %.2f" % (cls_name, error * 100, np.sum(indices), count)

    return error, count


def knn_predict(X_train, y_train, X_test, k=20):
    from sklearn.neighbors import KNeighborsClassifier

    neigh = KNeighborsClassifier(n_neighbors=k, algorithm='brute', metric='cosine')
    neigh.fit(X_train, y_train)

    return neigh.predict_proba(X_test)[:, 1]


def naive_bayes_predict(X_train, y_train, X_test):
    from sklearn.naive_bayes import MultinomialNB

    clf = MultinomialNB().fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


def random_forest_predict(X_train, y_train, X_test):
    from sklearn.ensemble import RandomForestClassifier 

    clf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


def svm_predict(X_train, y_train, X_test):
    from sklearn.svm import SVC

    clf = SVC(kernel='rbf', probability=True, cache_size=10000).fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


cutoff = 0.8

k_fold = KFold(X.shape[0], n_folds=10, shuffle=True)
for train, test in k_fold:
    X_train = X[train]
    y_train = labels[train]
    X_test = X[test] 
    y_test = labels[test]
    
    #proba = knn_predict(X_train, y_train, X_test, k=20)
    #test_high_prob_predictions(proba, y_test, cutoff, True, "KNN")

    proba = naive_bayes_predict(X_train, y_train, X_test)
    test_high_prob_predictions(proba, y_test, cutoff, True, "NB")


    proba = svm_predict(X_train[:500], y_train[:500], X_test)
    test_high_prob_predictions(proba, y_test, cutoff, True, "SVM")

    #proba = random_forest_predict(X_train, y_train, X_test)
    #test_high_prob_predictions(proba, y_test, cutoff, True, "RF")


    
