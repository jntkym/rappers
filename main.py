#!/usr/bin/env python
# -*- coding: utf8 -*-

import theano, theano.tensor as T
import numpy as np
import theano_lstm
import random
import numpy as np
import csv
import time
import sys
from utils import *
import logging
import argparse
import sys

from utils import *
import LM

reload(sys)
sys.setdefaultencoding('utf8')

logging.basicConfig(format='%(asctime)s/%(name)s[%(levelname)s]: %(message)s',level=logging.DEBUG)
logger = logging.getLogger(__name__)


theano.config.floatX = "float32"
rng = np.random.RandomState(1234)


def create_corpus(lyrics_file):
    logger.info("Preparing train set for extracting sentences ... ")
    data_corpus = []
    with open(lyrics_file) as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        for row in reader:
            sentences = row[2].decode("utf-8").split("<BR>")
            for sentence in sentences:
                items = []
                # first split on whitespce
                phrases = sentence.split(" ")
                for phrase in phrases:
                    words = get_cjk_characters(phrase)
                    items += words
                items.append(".")
                if len(items) > 0:
                    line = " ".join(items)
                    # yield items
                    data_corpus.append(line)
    logger.info(" Done ! ")
    return data_corpus

def pad_into_matrix(rows, padding=0):
    if len(rows) == 0:
        return np.array([0, 0], dtype=np.int32)
    lengths = map(len, rows)
    width = max(lengths)
    height = len(rows)
    mat = np.empty([height, width], dtype=rows[0].dtype)
    mat.fill(padding)
    for i, row in enumerate(rows):
        mat[i, 0:len(row)] = row
    return mat, list(lengths)




def main():


    parser = argparse.ArgumentParser(description="An LSTM language model")
    parser.add_argument('--train', help='Train NN using a train file', nargs=1)
    # parser.add_argument('-v', help='Specify a vocabulary file', nargs=1)
    parser.add_argument('-n', help='Number of epochs for training', nargs=1)
    parser.add_argument('--test', help='Run prediciton on a file', nargs=1)
    parser.add_argument('-m', help='Specify a file with model weights', nargs="+")

    opts = parser.parse_args()

    # DEFAULTS
    n_epochs = 10

    if opts.n:
        n_epochs = int(opts.n[0])

    if opts.train:

        logger.info("*** TRAINING MODE *** ")

        # generating corpus
        corpus = create_corpus(opts.train[0]) # the corpus is one sequence of characters per line
        with open("corpus","w") as f:
            for line in corpus:
                print >> f, line

        # building vocab
        vocab = Vocab()
        for sentence in corpus:
            vocab.add_words(sentence.split(" "))

        # generating the ind matrix
        numerical_lines = []
        for sentence in corpus:
            numerical_lines.append(vocab(sentence))
        idx_matrix, idx_vector_lengths = pad_into_matrix(numerical_lines)

        # construct model & theano functions:
        lm_model = LM.Model(
            input_size=10,
            hidden_size=10,
            vocab_size=len(vocab),
            stack_size=1, # make this bigger, but makes compilation slow
            celltype=LM.RNN # use RNN or LSTM
        )
        lm_model.vocab = vocab
        lm_model.stop_on(vocab.word2index["."])

        # training the model
        lm_model.train(n_epochs,idx_matrix,idx_vector_lengths)
        # lm_model.save("new_model.p")




    # if opts.test:
    #
    #     logger.info(" *** PREDICITON MODE *** ")
    #     logger.info("Preparing input ... ")
    #
    #     # # construct model & theano functions:
    #     # lm_model = LM.Model(
    #     #     input_size=10,
    #     #     hidden_size=10,
    #     #     vocab_size=len(vocab),
    #     #     stack_size=1,
    #     #     celltype=LM.LSTM
    #     # )
    #     # lm_model.vocab = vocab
    #     # lm_model.stop_on(vocab.word2index["."])
    #
    #
    #     logger.info("Start parsing ... ")
    #     result = parser.parse_input(input_corpus, theano=True)
    #
    #     # SAVING THE MODEL
    #     output = open(opts.parse[0]+"_parsed.txt",'w')
    #     logger.info("Parsing finished! Storing results in %s"%(output))
    #     for i in result:
    #         print >> output, i, "\n"
    #     output.close()


if __name__ == '__main__':
    main()
