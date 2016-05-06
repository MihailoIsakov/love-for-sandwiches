# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import bald_latin


def load_dataset():
    """
    Returns the balded comments, their labels, and scraped balded comments
    """
    # Load the labeled dataset
    corpus = open('dataset/lns_stemmed.txt', 'r').readlines()
    labels = open('dataset/lns_labels.txt', 'r').readlines()

    # Load the unlabeled comments
    scraped = open('dataset/stemmed.txt', 'r').readlines()

    # remove cyrillic and balden text
    corpus, labels = bald_latin.remove_cyrillic_comments(corpus, labels)
    corpus = bald_latin.remove_serbian_accents(corpus)
    scraped, _ = bald_latin.remove_cyrillic_comments(scraped, range(len(scraped)))
    scraped = bald_latin.remove_serbian_accents(scraped)

    # labels as a numpy array
    labels = np.array([int(float(x)) for x in labels])

    return corpus, labels, scraped


def build_new_corpus(original, o_labels, additions, a_labels):
    corpus = original + additions
    labels = o_labels + a_labels
    return corpus, labels


def build_vectorizer():
    croatian_stop_words = set([u"a",u"ako",u"ali",u"bi",u"bih",u"bila",u"bili",u"bilo",u"bio",u"bismo",u"biste",u"biti",u"bumo",u"da",u"do",u"duž",u"ga",u"hoće",u"hoćemo",u"hoćete",u"hoćeš",u"hoću",u"i",u"iako",u"ih",u"ili",u"iz",u"ja",u"je",u"jedna",u"jedne",u"jedno",u"jer",u"jesam",u"jesi",u"jesmo",u"jest",u"jeste",u"jesu",u"jim",u"joj",u"još",u"ju",u"kada",u"kako",u"kao",u"koja",u"koje",u"koji",u"kojima",u"koju",u"kroz",u"li",u"me",u"mene",u"meni",u"mi",u"mimo",u"moj",u"moja",u"moje",u"mu",u"na",u"nad",u"nakon",u"nam",u"nama",u"nas",u"naš",u"naša",u"naše",u"našeg",u"ne",u"nego",u"neka",u"neki",u"nekog",u"neku",u"nema",u"netko",u"neće",u"nećemo",u"nećete",u"nećeš",u"neću",u"nešto",u"ni",u"nije",u"nikoga",u"nikoje",u"nikoju",u"nisam",u"nisi",u"nismo",u"niste",u"nisu",u"njega",u"njegov",u"njegova",u"njegovo",u"njemu",u"njezin",u"njezina",u"njezino",u"njih",u"njihov",u"njihova",u"njihovo",u"njim",u"njima",u"njoj",u"nju",u"no",u"o",u"od",u"odmah",u"on",u"ona",u"oni",u"ono",u"ova",u"pa",u"pak",u"po",u"pod",u"pored",u"prije",u"s",u"sa",u"sam",u"samo",u"se",u"sebe",u"sebi",u"si",u"smo",u"ste",u"su",u"sve",u"svi",u"svog",u"svoj",u"svoja",u"svoje",u"svom",u"ta",u"tada",u"taj",u"tako",u"te",u"tebe",u"tebi",u"ti",u"to",u"toj",u"tome",u"tu",u"tvoj",u"tvoja",u"tvoje",u"u",u"uz",u"vam",u"vama",u"vas",u"vaš",u"vaša",u"vaše",u"već",u"vi",u"vrlo",u"za",u"zar",u"će",u"ćemo",u"ćete",u"ćeš",u"ću",u"što"])

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


def test_high_prob_predictions(proba, y, min_cert, prnt=False, cls_name=""):
    """
    Tests only the probabilities above min_cert.
    """
    assert len(proba) == len(y)
    indices = np.logical_or(proba >= min_cert, proba <= 1 - min_cert)

    # classification error for comments with high certiainty
    error = np.mean(np.round(proba[indices]) == np.round(y[indices]))
    # number of comments classified with good certainty
    count = float(np.sum(indices)) / len(indices) * 100

    if prnt:
        print "%s --- accuracy: %.2f, classified: %d, perc classified: %.2f" % (cls_name, error * 100, np.sum(indices), count)

    return error, count


def _prune_comments(proba, threshold=0.9):
    """
    Takes the label probabilities,
    returns the indices of comments that should be added to the training set,
    and their labels 
    """
    proba = np.array(proba)
    # comments that fit in the [0, 1-threshold] u [threshold, 1] range
    fits = np.logical_or(proba >= threshold, proba =< 1 - threshold)  
    selected = np.argwhere(fits)

    # get the labels of the selected comments
    labels = np.array([int(np.round(proba[x])) for x in selected])
    return selected, labels


def label_candidates(corpus, labels, unlabeled):
    """
    Take the labeled corpus and the unlabeled dataset,
    fit the vectorizer on the corpus, transform the unlabeled comments, 
    train the NB on the corpus and labels,
    process the unlabeled comments, 
    return comments that can be classified with a high certainty.
    """
    # build tf-idf vectorizer, fit on the corpus, transform unlabeled comments
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(corpus).todense()
    X_unlabeled = vectorizer.transform(unlabeled).todense()

    # create and fit the Naive Bayes classifier on the TF-IDF'd corpus and labels
    nb = MultinomialNB().fit(X_train, labels)

    # predict labels for the unlabeled comments,
    proba = nb.predict_proba(X_unlabeled)
    best_indices, new_labels = _prune_comments(proba)
    
    return best_indices, new_lables
