#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
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
import logging
import argparse
import numpy as np
import theano

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM

import LM

reload(sys)
sys.setdefaultencoding('utf8')

logging.basicConfig(format='%(asctime)s/%(name)s[%(levelname)s]: %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# theano.config.floatX = "float32"
rng = np.random.RandomState(1234)


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


def main():
    parser = argparse.ArgumentParser(description="An LSTM language model")
    parser.add_argument('--train', help='Train NN using a train file', nargs=1)
    parser.add_argument('-n', help='Number of epochs for training', nargs=1)
    parser.add_argument('--test', help='Run prediciton on a file', nargs=1)
    parser.add_argument('-m', help='Specify a file with model weights', nargs="+")

    opts = parser.parse_args()

    # DEFAULTS
    n_epochs = 30
    model = "keras"
    save_model = False

    maxlen = 20
    step = 3

    if opts.n:
        n_epochs = int(opts.n[0])

    if opts.train:
        t0 = time.time()
        logger.info("*** TRAINING MODE *** ")

        if model == "keras":
            logger.debug("Using keras model")

            X, y, text, char_indices, indices_char = preprocess.prepare_NN_input(opts.train[0],
                                                                                 model="keras",
                                                                                 savepath="data/NN_input.txt",
                                                                                 maxlen=maxlen,
                                                                                 step=step)

            vocab_size = len(char_indices.keys())
            text_size = len(text)

            # build the model: 2 stacked LSTM
            print('Building model...')
            model = Sequential()
            model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, vocab_size)))
            model.add(Dropout(0.2))
            model.add(LSTM(512, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(vocab_size))
            model.add(Activation('softmax'))

            model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

            # train the model, output generated text after each iteration
            for iteration in range(1, n_epochs):
                print()
                print('-' * 50)
                print('Iteration', iteration)
                model.fit(X, y, batch_size=128, nb_epoch=1)

                start_index = random.randint(0, text_size - maxlen - 1)

                for diversity in [0.2, 0.5, 1.0, 1.2]:
                    print()
                    print('----- diversity:', diversity)

                    generated = ''
                    sentence = text[start_index: start_index + maxlen]
                    generated += sentence
                    print('----- Generating with seed: "' + sentence + '"')
                    sys.stdout.write(generated)

                    for iteration in range(400):
                        x = np.zeros((1, maxlen, vocab_size))
                        for t, char in enumerate(sentence):
                            x[0, t, char_indices[char]] = 1.

                        preds = model.predict(x, verbose=0)[0]
                        next_index = sample(preds, diversity)
                        next_char = indices_char[next_index]

                        generated += next_char
                        sentence = sentence[1:] + next_char

                        sys.stdout.write(next_char)
                        sys.stdout.flush()
                    print()

            if save_model:
                model.save_weights('keras_model_weights.h5')

        elif model == "theano":

            logger.debug("Using theano model")

            # generating cleaned lyrics corpus from crawled data
            text, char_indices, indices_char, x_tr, y_tr, x_te, y_te = preprocess.prepare_NN_input(opts.train[0],
                                                                                                   model="theano",
                                                                                                   savepath="data/NN_input.txt",
                                                                                                   maxlen=maxlen,
                                                                                                   step=step)

            vocab_size = len(char_indices.keys())
            text_size = len(text)

            n_u = len(char_indices.keys())
            n_h = 10
            n_y = len(char_indices.keys())
            np.random.seed(0)

            model = LM.RNN(n_u=n_u, n_h=n_h, n_y=n_y,
                           activation='lstm', output_type='softmax',
                           learning_rate=0.001, learning_rate_decay=0.999,
                           L1_reg=0, L2_reg=0,
                           initial_momentum=0.5, final_momentum=0.9,
                           momentum_switchover=5,
                           n_epochs=n_epochs)

            model.build_train(text, char_indices, indices_char,
                              x_tr, y_tr, x_te, y_te, save=save_model)

            logger.info("Done training the model ! Elapsed time: %f" % (time.time() - t0))


if __name__ == '__main__':
    main()
