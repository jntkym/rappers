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
import argparse

reload(sys)
sys.setdefaultencoding('utf8')
logger = logging.getLogger(__name__)

# pattern for removing redundunt spaces
space_pat = re.compile(u'\s\s+', re.U)


# # pattern for removing English - not used now
# eng_words_pat = re.compile(u'[A-Za-z]*',re.U)

# aux 1.1+
def translate_non_alphanumerics(to_translate, translate_to=None):
    not_letters_or_digits = u'[!&#%\"\'()_`{※+,』\|}~?...…「〜＞ｒ（）＜｀！」？＿％・@＠”’"：；＋ー！。。。、＿・_ _『 □**＊-\.\/:;<=>△?@\[\]\^'
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
            sentences = row[2].strip().split(u"<BR>")
            for sentence in sentences:
                sentence = unicode(sentence)
                sentence = translate_non_alphanumerics(sentence)
                sentence = space_pat.sub(u' ', sentence)

                # delete English
                # sentence = eng_words_pat.sub(u'', sentence).split(u"\s")

                # sentence = sentence.split(u'')
                # sentence.append(u".")
                # sentence += u'.'

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
    unk_pat = re.compile(u"/補助記号/UNK")
    slash_pat = re.compile(ur"\\")

    with codecs.open("data/kytea_out.txt", 'UTF-8') as f:
        for line in f:
            line = line.decode(encoding="utf-8").strip()
            line = unk_pat.sub(u"", line)
            line = slash_pat.sub(u"", line)

            triplets = line.split(u" ")  # take a look at Kytea output: https://github.com/chezou/Mykytea-python
            seq = []
            for item in triplets:
                try:
                    # hir = item.split(u"/")[2]
                    # if hir != "UNK":
                    hir = item.split(u"/")[0]
                    if hir != "\\":
                        seq.append(hir)
                except IndexError:
                    continue

            candidate_line = unicodedata.normalize("NFKC", u" ".join(seq))
            candidate_line = re.sub(u"[A-Za-z]", u"", candidate_line)
            candidate_line = re.sub(u"\s+", u"", candidate_line)
            candidate_line = re.sub(u"\d+", u"5", candidate_line)

            if len(candidate_line) > 10:
                pron.append(candidate_line)


    NN_input = u"\n".join(pron)
    return NN_input


# main function - creates input for the NN
def clean_corpus(crawled_lyrics_file, model="keras", savepath=None):
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

            # if model == "keras":
            #
            #     chars = set(text)
            #     vocab_size = len(chars)
            #     text_size = len(text)
            #
            #     print('corpus length:', len(text))
            #     print('total chars:', vocab_size)
            #
            #     char_indices = dict((c, i) for i, c in enumerate(chars))
            #     indices_char = dict((i, c) for i, c in enumerate(chars))
            #
            #     # cut the text in semi-redundant sequences of maxlen characters
            #
            #     sentences = []
            #     next_chars = []
            #     for i in range(0, len(text) - maxlen, step):
            #         sentences.append(text[i: i + maxlen])
            #         next_chars.append(text[i + maxlen])
            #     print('nb sequences:', len(sentences))
            #
            #     print('Vectorization...')
            #     X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
            #     y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
            #     for i, sentence in enumerate(sentences):
            #         for t, char in enumerate(sentence):
            #             X[i, t, char_indices[char]] = 1
            #         y[i, char_indices[next_chars[i]]] = 1
            #
            #     return X, y, text, char_indices, indices_char

            # # the following lines are for theano model only


def process_juman_output(juman_outfile):
    corpus = []
    daihyou_vocab = {}

    with open(juman_outfile) as csvfile:
        reader = csv.reader(csvfile, delimiter=str(u" "))
        sent = []

        for line in reader:
            if line[0] == u"@":
                continue

            if line[0] == u"EOS":
                corpus.append(u" ".join(sent))
                sent=[]
                continue

            if line[11] != "NIL":
                value = line[11]
                value = re.sub("代表表記:", u"", value,re.U)
                value = value.split(u"/")[0]
            else:
                value = line[0]

            key = line[0]
            daihyou_vocab[key] = value
            sent.append(key)
        corpus = u"\n".join(corpus)


    print "All in all unique lemmas: %d" %(len(daihyou_vocab.values()))

    # save a txt corpus file
    with open("data/string_corpus.txt","w") as f:
        for line in corpus.split(u"\n"):
            print >> f, line

    # save a vocabulary
    with open("data/daihyou_vocab.p", "w") as vocabfile:
        pickle.dump(daihyou_vocab, vocabfile)

    print "cleaning datadir ..."
    subprocess.call('rm -f ./data/clean_corpus.txt ./data/kytea_out.txt ./data/cleaned_lyrics.txt',
                    shell=True)

def main():
    parser = argparse.ArgumentParser(description="An LSTM language model")
    parser.add_argument('-juman', help='Preprocess juman file', nargs=1)
    parser.add_argument('-crawl', help='Preprocess crawled data', nargs=1)

    opts = parser.parse_args()

    if opts.crawl:
        print "Processing crawled data ..."
        clean_corpus(opts.crawl[0], savepath="data/clean_corpus.txt")
        print "Done"

    if opts.juman:
        print "Processing juman output ... "
        process_juman_output(opts.juman[0])
        print "Done"

if __name__ == '__main__':
    main()
