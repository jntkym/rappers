#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import csv
import time
import sys
import utils
import preprocess


from collections import OrderedDict
import string
import subprocess
import codecs

import numpy as np
import theano
from passage.models import RNN
from passage.updates import NAG, Regularizer
from passage.layers import Embedding, LstmRecurrent, Dense
from passage.utils import load, save

import logging
import argparse



reload(sys)
sys.setdefaultencoding('utf8')

logging.basicConfig(format='%(asctime)s/%(name)s[%(levelname)s]: %(message)s',level=logging.DEBUG)
logger = logging.getLogger(__name__)


theano.config.floatX = "float32"
rng = np.random.RandomState(1234)


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
    save_corpus=True

    if opts.n:
        n_epochs = int(opts.n[0])

    if opts.train:

        logger.info("*** TRAINING MODE *** ")
        logger.info("Preparing data ...")
        # generating cleaned lyrics corpus from crawled data
        x_tr, y_tr, x_te,y_te = preprocess.prepare_NN_input(opts.train[0], save=True)


        layers = [
            Embedding(size=28),
            LstmRecurrent(size=512, p_drop=0.2),
            LstmRecurrent(size=512, p_drop=0.2),
            Dense(size=10, activation='softmax', p_drop=0.5)
        ]

        # #A bit of l2 helps with generalization, higher momentum helps convergence
        # updater = NAG(momentum=0.95, regularizer=Regularizer(l2=1e-4))
        #
        # #Linear iterator for real valued data, cce cost for softmax
        # model = RNN(layers=layers, updater=updater, iterator='linear', cost='cce')
        # model.fit(trX, trY, n_epochs=20)
        #
        # tr_preds = model.predict(trX[:len(teY)])
        # te_preds = model.predict(teX)
        #
        # tr_acc = np.mean(trY[:len(teY)] == np.argmax(tr_preds, axis=1))
        # te_acc = np.mean(teY == np.argmax(te_preds, axis=1))
        #
        # # Test accuracy should be between 98.9% and 99.3%
        # print 'train accuracy', tr_acc, 'test accuracy', te_acc



if __name__ == '__main__':
    main()
