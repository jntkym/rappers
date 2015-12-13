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
import unicodedata

reload(sys)
sys.setdefaultencoding('utf8')
logger = logging.getLogger(__name__)

# pattern for removing redundunt spaces
space_pat = re.compile(u'\s\s+', re.U)


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

                if len(sentence) > 1:
                    data_corpus.append(sentence)

    logger.info(" Done cleaning crawled data! ")

    # saving the corpus
    with codecs.open("data/cleaned_lyrics.txt", "w", 'UTF-8') as f:
        f.write("\n".join(data_corpus))


# aux 2
def create_corpus(crawled_lyrics_file, save=False):
    """
    Load cleaned crawled corpus from local folder, feed to Kytea, get the output.
    Then laod the output and do post-processing.

    :param crawled_lyrics_file:
    :param save:
    :return: clean_corpus.txt file which is fed to LSTM
    """

    # generating cleaned lyrics corpus from crawled data
    clean_lyrics(crawled_lyrics_file)  # the corpus is one sequence of characters per line
    subprocess.call('kytea < ./data/cleaned_lyrics.txt > ./data/kytea_out.txt', shell=True)  # processing with kytea
    logger.info(" Done kytea processing! ")

    pron = []
    unk_pat = re.compile(u"/.*UNK")
    slash_pat = re.compile(ur"\\")

    with codecs.open("data/kytea_out.txt", 'UTF-8') as f:
        for line in f:
            line = line.decode(encoding="utf-8").strip()
            unk_pat.sub(u"", line)
            slash_pat.sub(u"", line)
            triplets = line.split(u" ")  # take a look at Kytea output: https://github.com/chezou/Mykytea-python
            seq = []
            for item in triplets:
                try:
                    hir = item.split(u"/")[2]
                    if hir != "UNK":
                        seq.append(hir)
                except IndexError:
                    continue

            if len(seq) > 3:
                pron.append(u" ".join(seq))
            else:
                pron.append(u"\n")

    NN_input = unicodedata.normalize("NFKC",u"\n".join(pron))
    NN_input = re.sub(u"\d+",u"5",NN_input)

    return NN_input


# main function - creates input for the NN
def prepare_NN_input(crawled_lyrics_file, model="keras", savepath=None, maxlen=20, step=3):
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

    logger.info("Preparing data ...")
    text = create_corpus(crawled_lyrics_file, save=False).lower()

    if savepath != None:
        with open(savepath, "w") as f:
            f.write(text)
            logger.info(" Corpus saved into ----->%s " % (savepath))

    if model == "keras":

        chars = set(text)
        vocab_size = len(chars)
        text_size = len(text)

        print('corpus length:', len(text))
        print('total chars:', vocab_size)

        char_indices = dict((c, i) for i, c in enumerate(chars))
        indices_char = dict((i, c) for i, c in enumerate(chars))

        # cut the text in semi-redundant sequences of maxlen characters

        sentences = []
        next_chars = []
        for i in range(0, len(text) - maxlen, step):
            sentences.append(text[i: i + maxlen])
            next_chars.append(text[i + maxlen])
        print('nb sequences:', len(sentences))

        print('Vectorization...')
        X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                X[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1

        return X, y, text, char_indices, indices_char

    # the following lines are for theano model only
    elif model == "theano":

        chars = set(text)
        # chars.add("EOS")
        char_indices = dict((c, i) for i, c in enumerate(chars))
        indices_char = dict((i, c) for i, c in enumerate(chars))

        print('corpus length: ', len(text))
        print('total chars:', len(chars))

        # cut the text into sequences of characters

        sentences = []
        next_chars = []
        for i in range(0, len(text) - maxlen, step):
            sentences.append(text[i: i + maxlen])  # sequence of 20 chars
            next_chars.append(text[i + 1: i + maxlen + 1])  # sequence of 20 chars
        print('nb sequences:', len(sentences))

        logger.info('Vectorization...')

        X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
        y = np.zeros((len(sentences), maxlen), dtype=np.int)

        for sent_idx, sentence in enumerate(sentences):

            # print "\t*sanity check *"
            # print "sentence: %s" %(sentence)

            for ch_idx, char in enumerate(sentence):
                X[sent_idx, ch_idx, char_indices[char]] = 1
                y[sent_idx, ch_idx] = char_indices[next_chars[sent_idx][ch_idx]]

                # print "current character - %s, next character - %s"%(char, next_chars[sent_idx][ch_idx])

                # print "\tsanity check over*"
                # time.sleep(5)

        split_ratio = 0.8
        x_train = X[:len(sentences) * split_ratio, :, :]
        y_train = y[:len(sentences) * split_ratio, :]

        x_test = X[len(sentences) * split_ratio:, :, :]
        y_test = y[len(sentences) * split_ratio:, :]

        logger.info(" Done generating NN input! ")

        return text, char_indices, indices_char, x_train, y_train, x_test, y_test
