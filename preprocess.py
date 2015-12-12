#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import time
import sys
import logging
import sys
import subprocess
import codecs
from utils import *


reload(sys)
sys.setdefaultencoding('utf8')
logger = logging.getLogger(__name__)

# pattern for removing redundunt spaces
space_pat=re.compile(u'\s\s+', re.U)

# # pattern for removing English - not used now
# eng_words_pat = re.compile(u'[A-Za-z]*',re.U)

# aux 1.1+
def translate_non_alphanumerics(to_translate, translate_to=None):
    not_letters_or_digits = u'!"&#%\'()*＊+,-./:;<=>?@[\]^_`{|}~...…「〜＞ｒ（）＜｀」％＿・'
    translate_table = dict((ord(char), translate_to) for char in not_letters_or_digits)
    return to_translate.translate(translate_table)

# aux 1+
def clean_lyrics(lyrics_file):
    """
    Take crawled data and do some simple cleaning on it
    :param lyrics_file: file from Otani san - crawled data
    :return: cleaned data, which will be fed to Kytea
    """
    data_corpus = []
    with open(lyrics_file) as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        for row in reader:
            sentences = row[2].decode("utf-8").strip().split(u"<BR>")
            for sentence in sentences:

                sentence = translate_non_alphanumerics(sentence)
                sentence = space_pat.sub(u' ', sentence.strip())

                # delete English
                # sentence = eng_words_pat.sub(u'', sentence).split(u"\s")

                # sentence = sentence.split(u'')
                # sentence.append(u".")
                sentence += u'.'

                if len(sentence)>1:
                    data_corpus.append(sentence)

    logger.info(" Done cleaning crawled data! ")

    # saving the corpus
    with codecs.open("cleaned_lyrics.txt","w",'UTF-8') as f:
        f.write("\n".join(data_corpus))

# aux 2
def create_corpus(crawled_lyrics_file,save=False):
    """
    Load cleaned crawled corpus from local folder, feed to Kytea, get the output.
    Then laod the output and do post-processing.

    :param crawled_lyrics_file:
    :param save:
    :return: clean_corpus.txt file which is fed to LSTM
    """

    # generating cleaned lyrics corpus from crawled data
    clean_lyrics(crawled_lyrics_file) # the corpus is one sequence of characters per line
    subprocess.call('kytea < cleaned_lyrics.txt > kytea_out.txt', shell=True) # processing with kytea
    logger.info(" Done kytea processing! ")

    pron=[]
    unk_pat = re.compile(u"/.*UNK")
    slash_pat = re.compile(ur"\\")

    with codecs.open("kytea_out.txt",'UTF-8') as f:
        for line in f:
            line=line.decode(encoding="utf-8").strip()
            unk_pat.sub(u"",line)
            slash_pat.sub(u"",line)
            triplets=line.split(u" ") # take a look at Kytea output: https://github.com/chezou/Mykytea-python
            seq = []
            for item in triplets:
                try:
                    hir = item.split(u"/")[2]
                    if hir != "UNK":
                        seq.append(hir)
                except IndexError:
                    continue

            if len(seq)>3:
                pron.append(u" ".join(seq))
            else:
                pron.append(u"\n")

    NN_input = u"\n".join(pron)

    return NN_input

# main function - creates input for the NN
def prepare_NN_input(crawled_lyrics_file,savepath=None):

    """
    Prepares 4 matrices: x_train, y_train, x_test, y_test.
    X matrices are of size N x len(seq) x V:
    - N is the number of samples,
    - len(seq) is the length of the time series (= number of timestamps),
    - V is the vocab size.

    Y matirces are of size N x V

    :param filename: cleaned_corpus
    :return: 4 matrices
    """

    text = create_corpus(crawled_lyrics_file, save=False).lower()


    if savepath:
        with open("clean_corpus_jp.txt","w") as f:
            f.write(text)
            logger.info(" Corpus saved into ----->%s "%(savepath))

    chars = set(text)
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    logger.info('corpus length:', len(text))
    logger.info('total chars:', len(chars))

    # cut the text in sequences of characters
    maxlen = 20
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    logger.info('nb sequences:', len(sentences))

    logger.info('Vectorization...')
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    split_ratio = 0.8
    x_train = X[:len(sentences)*split_ratio,:,:]
    y_train = y[:len(sentences)*split_ratio,:]

    x_test = X[len(sentences)*split_ratio:,:,:]
    y_test = y[len(sentences)*split_ratio:,:]

    logger.info(" Done generating NN input! ")

    return x_train,y_train, x_test,y_test