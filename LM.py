#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""

Recurrent Neural Network containing LSTM and GRU hidden layer
Code provided by Mohammad Pezeshki - Nov. 2014 - Universite de Montreal
This code is distributed without any warranty, express or implied.
Taken from: https://github.com/mohammadpz/Recurrent-Neural-Networks/blob/master/RNN_with_gating.py

"""
import os
import datetime
import sys
from time import time
import codecs
import logging
import random
import numpy as np
import theano
import theano.tensor as T
import time

import lstm_gru
from collections import OrderedDict
import cPickle as pickle

mode = theano.Mode(linker='cvm')  # the runtime algo to execute the code is in c
logger = logging.getLogger(__name__)
# plt.ion()

"""
What we have in this class:

    Model structure parameters:
        n_u : length of input layer vector in each time-step
        n_h : length of hidden layer vector in each time-step
        n_y : length of output layer vector in each time-step
        activation : type of activation function used for hidden layer
                     can be: sigmoid, tanh, relu, lstm, or gru
        output_type : type of output which could be `real`, `binary`, or `softmax`

    Parameters to be learned:
        W_uh : weight matrix from input to hidden layer
        W_hh : recurrent weight matrix from hidden to hidden layer
        W_hy : weight matrix from hidden to output layer
        b_h : biases vector of hidden layer
        b_y : biases vector of output layer
        h0 : initial values for the hidden layer

    Learning hyper-parameters:
        learning_rate : learning rate which is not constant
        learning_rate_decay : learning rate decay :)
        L1_reg : L1 regularization term coefficient
        L2_reg : L2 regularization term coefficient
        initial_momentum : momentum value which we start with
        final_momentum : final value of momentum
        momentum_switchover : on which `epoch` should we switch from
                              initial value to final value of momentum
        n_epochs : number of iterations

    Inner class variables:
        self.x : symbolic input vector
        self.y : target output
        self.y_pred : raw output of the model
        self.p_y_given_x : output after applying sigmoid (binary output case)
        self.y_out : round (0,1) for binary and argmax (0,1,...,k) for softmax
        self.loss : loss function (MSE or CrossEntropy)
        self.predict : a function returns predictions which is type is related to output type
        self.predict_proba : a function returns predictions probabilities (binary and softmax)
    
    build_train function:
        train_set_x : input of network
        train_set_y : target of network
        index : index over each of training sequences (NOT the number of time-steps)
        lr : learning rate
        mom : momentum
        cost : cost function value
        compute_train_error : a function compute error on training
        gparams : Gradients of model parameters
        updates : updates which should be applied to parameters
        train_model : a function that returns the cost, but 
                      in the same time updates the parameter
                      of the model based on the rules defined
                      in `updates`.
        
"""


class RNN(object):
    def __init__(self, n_u, n_h, n_y, activation, output_type,
                 learning_rate, learning_rate_decay, L1_reg, L2_reg,
                 initial_momentum, final_momentum, momentum_switchover,
                 n_epochs):

        self.n_u = int(n_u)  # dimensionality of the input VECTORS (essentially - size of the vocabulary)
        self.n_h = int(n_h)  # dimensionality of the hidden states
        self.n_y = int(n_y)  # dimensionality of the output vectors

        if activation == 'tanh':
            self.activation = T.tanh
        elif activation == 'sigmoid':
            self.activation = T.nnet.sigmoid
        elif activation == 'relu':
            self.activation = lambda x: x * (x > 0)  # T.maximum(x, 0)
        elif activation == 'lstm':
            self.lstm = lstm_gru.LSTM(n_u, n_h)
            self.activation = self.lstm.lstm_as_activation_function
        elif activation == 'gru':
            self.gru = lstm_gru.GRU(n_u, n_h)
            self.activation = self.gru.gru_as_activation_function
        else:
            raise NotImplementedError

        self.output_type = output_type
        self.learning_rate = float(learning_rate)
        self.learning_rate_decay = float(learning_rate_decay)
        self.L1_reg = float(L1_reg)
        self.L2_reg = float(L2_reg)
        self.initial_momentum = float(initial_momentum)
        self.final_momentum = float(final_momentum)
        self.momentum_switchover = int(momentum_switchover)
        self.n_epochs = int(n_epochs)

        # input which is `x`
        self.x = T.matrix()

        # Note that some variables below are not used when
        # the activation function is LSTM or GRU. But we simply
        # don't care because theano optimize this for us.

        # Weights are initialized from an uniform distribution
        self.W_uh = theano.shared(value=np.asarray(
            np.random.uniform(
                size=(n_u, n_h),
                low=-.01, high=.01),
            dtype=theano.config.floatX),
            name='W_uh')

        self.W_hh = theano.shared(value=np.asarray(
            np.random.uniform(
                size=(n_h, n_h),
                low=-.01, high=.01),
            dtype=theano.config.floatX),
            name='W_hh')

        self.W_hy = theano.shared(value=np.asarray(
            np.random.uniform(
                size=(n_h, n_y),
                low=-.01, high=.01),
            dtype=theano.config.floatX),
            name='W_hy')

        # initial value of hidden layer units are set to zero
        self.h0 = theano.shared(value=np.zeros(
            (n_h,),
            dtype=theano.config.floatX),
            name='h0')

        self.c0 = theano.shared(value=np.zeros(
            (n_h,),
            dtype=theano.config.floatX),
            name='c0')

        # biases are initialized to zeros
        self.b_h = theano.shared(value=np.zeros(
            (n_h,),
            dtype=theano.config.floatX),
            name='b_h')

        self.b_y = theano.shared(value=np.zeros(
            (n_y,),
            dtype=theano.config.floatX),
            name='b_y')
        # That's because when it is lstm or gru, parameters are different
        if activation == 'lstm':
            # Note that `+` here is just a concatenation operator
            self.params = self.lstm.params + [self.W_hy, self.h0, self.b_y]
        elif activation == 'gru':
            self.params = self.gru.params + [self.W_hy, self.h0, self.b_y]
        else:
            self.params = [self.W_uh, self.W_hh, self.W_hy, self.h0,
                           self.b_h, self.b_y]

        # Initial value for updates is zero matrix.
        self.updates = OrderedDict()
        for param in self.params:
            self.updates[param] = theano.shared(
                value=np.zeros(
                    param.get_value(
                        borrow=True).shape,
                    dtype=theano.config.floatX),
                name='updates')

        # Default value of c_tm1 is None since we use it just when we have LSTM units
        def recurrent_fn(u_t, h_tm1, c_tm1=None):
            # that's because LSTM needs both u_t and h_tm1 to compute gates
            if activation == 'lstm':
                h_t, c_t = self.activation(u_t, h_tm1, c_tm1)
            elif activation == 'gru':
                h_t = self.activation(u_t, h_tm1)
                # In this case, we don't need c_t; but we need to return something.
                # On the other hand, we cnannot return None. Thus,
                # To use theano optimazation features, let's just return h_t.
                c_t = h_t  # Just to get rid of c_t
            else:
                h_t = self.activation(T.dot(u_t, self.W_uh) + \
                                      T.dot(h_tm1, self.W_hh) + \
                                      self.b_h)
                # Read above comment
                c_t = h_t  # Just to get rid of c_t

            y_t = T.dot(h_t, self.W_hy) + self.b_y
            return h_t, c_t, y_t

        # Iteration over the first dimension of a tensor which is TIME in our case.
        # recurrent_fn doesn't use y in the computations, so we do not need y0 (None)
        # scan returns updates too which we do not need. (_)
        [self.h, self.c, self.y_pred], _ = theano.scan(fn=recurrent_fn,
                                                       sequences=self.x,
                                                       outputs_info=[self.h0, self.c0,
                                                                     None])  # this should correspond to return type of fn

        # L1 norm
        self.L1 = abs(self.W_uh.sum()) + \
                  abs(self.W_hh.sum()) + \
                  abs(self.W_hy.sum())

        # square of L2 norm
        self.L2_sqr = (self.W_uh ** 2).sum() + \
                      (self.W_hh ** 2).sum() + \
                      (self.W_hy ** 2).sum()

        # Loss function is different for different output types
        # defining function in place is so easy! : lambda input: expression
        if self.output_type == 'real':
            self.y = T.matrix(name='y', dtype=theano.config.floatX)
            self.loss = lambda y: self.mse(y)  # y is input and self.mse(y) is output
            self.predict = theano.function(inputs=[self.x, ],
                                           outputs=self.y_pred,
                                           mode=mode)

        elif self.output_type == 'binary':
            self.y = T.matrix(name='y', dtype='int32')
            self.p_y_given_x = T.nnet.sigmoid(self.y_pred)
            self.y_out = T.round(self.p_y_given_x)  # round to {0,1}
            self.loss = lambda y: self.nll_binary(y)
            self.predict_proba = theano.function(inputs=[self.x, ],
                                                 outputs=self.p_y_given_x,
                                                 mode=mode)
            self.predict = theano.function(inputs=[self.x, ],
                                           outputs=self.y_out,
                                           mode=mode)

        elif self.output_type == 'softmax':
            self.y = T.vector(name='y', dtype='int32')
            self.p_y_given_x = T.nnet.softmax(self.y_pred)
            self.y_out = T.argmax(self.p_y_given_x, axis=-1)
            self.loss = lambda y: self.nll_multiclass(y)
            self.predict_proba = theano.function(inputs=[self.x, ],
                                                 outputs=self.p_y_given_x,
                                                 mode=mode)

            self.predict = theano.function(inputs=[self.x, ],
                                           outputs=self.y_out,  # y-out is calculated by applying argmax
                                           mode=mode)
        else:
            raise NotImplementedError

        # Just for tracking training error for Graph 3
        self.errors = []

    def mse(self, y):
        # mean is because of minibatch
        return T.mean((self.y_pred - y) ** 2)

    def nll_binary(self, y):
        # negative log likelihood here is cross entropy
        return T.mean(T.nnet.binary_crossentropy(self.p_y_given_x, y))

    def nll_multiclass(self, y):  # y is a vector of values (length is the timeseries' length)
        # notice to [  T.arange(y.shape[0])  ,  y  ]
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def sample(a, temperature=1.0):
        # helper function to sample an index from a probability array
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        return np.argmax(np.random.multinomial(1, a, 1))

    # X_train, Y_train, X_test, and Y_test are numpy arrays
    def build_train(self, text, char_indices, indices_char,
                    X_train, Y_train, X_test=None, Y_test=None, save=False):

        # self.ch_idx = char_indices
        # self.idx_ch = indices_char

        train_set_x = theano.shared(np.asarray(X_train, dtype=theano.config.floatX))
        train_set_y = theano.shared(np.asarray(Y_train, dtype=theano.config.floatX))
        if self.output_type in ('binary', 'softmax'):
            train_set_y = T.cast(train_set_y, 'int32')

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print 'Building model ...'

        index = T.lscalar('index')  # index to a case
        # learning rate (may change)
        lr = T.scalar('lr', dtype=theano.config.floatX)
        mom = T.scalar('mom', dtype=theano.config.floatX)  # momentum


        # Note that we use cost for training
        # But, compute_train_error for just watching and printing
        cost = self.loss(self.y) \
               + self.L1_reg * self.L1 \
               + self.L2_reg * self.L2_sqr

        # We don't want to pass whole dataset every time we use this function.
        # So, the solution is to put the dataset in the GPU as `givens`.
        # And just pass index to the function each time we have some input.
        compute_train_error = theano.function(inputs=[index, ],
                                              outputs=self.loss(self.y),
                                              givens={
                                                  self.x: train_set_x[index],
                                                  self.y: train_set_y[index]},
                                              mode=mode)

        # Gradients of cost wrt. [self.W, self.W_in, self.W_out,
        # self.h0, self.b_h, self.b_y] using BPTT.
        gparams = []
        for param in self.params:
            gparams.append(T.grad(cost, param))

        # zip just concatenate two lists
        updates = OrderedDict()
        for param, gparam in zip(self.params, gparams):
            weight_update = self.updates[param]
            upd = mom * weight_update - lr * gparam
            updates[weight_update] = upd
            updates[param] = param + upd

        # compiling a Theano function `train_model` that returns the
        # cost, but in the same time updates the parameter of the
        # model based on the rules defined in `updates`
        train_model = theano.function(inputs=[index, lr, mom],
                                      outputs=cost,
                                      updates=updates,
                                      givens={
                                          self.x: train_set_x[index],  # [:, batch_start:batch_stop]
                                          self.y: train_set_y[index]},
                                      mode=mode)

        ###############
        # TRAIN MODEL #
        ###############
        print 'Training model ...'
        epoch = 0
        maxlen = 20
        n_train = train_set_x.get_value(borrow=True).shape[0]

        while (epoch < self.n_epochs):
            epoch = epoch + 1

            print('-' * 50)

            for idx in xrange(n_train):
                effective_momentum = self.final_momentum if epoch > self.momentum_switchover \
                    else self.initial_momentum

                # the training is essentially online - one example sentence per iteration
                example_cost = train_model(idx,
                                           self.learning_rate,
                                           effective_momentum)
                if idx % 1000 == 0:
                    logger.info("Training on %d example" % (idx))


            # compute loss on training set
            train_losses = [compute_train_error(i)
                            for i in xrange(n_train)]
            this_train_loss = np.mean(train_losses)
            self.errors.append(this_train_loss)

            print('epoch %i, train loss %f ''lr: %f' % (epoch, this_train_loss, self.learning_rate))

            # sample predictions on the test set
            start_index = random.randint(0, len(text) - maxlen - 1)

            for diversity in [0.2, 0.5, 1.0, 1.2]:

                print('----- diversity:', diversity)

                generated = ''
                sentence = text[start_index: start_index + maxlen]
                generated += sentence
                print('----- Generating with seed: "' + sentence + '"')
                sys.stdout.write(generated)

                for iteration in range(400):
                    x = np.zeros((maxlen, len(char_indices.keys())))
                    for t, char in enumerate(sentence):
                        x[t, char_indices[char]] = 1.

                    # this gives a (vocab_size,) vector with probs for each
                    # character in the vocabulary at the last timestamp
                    # preds = np.squeeze(self.predict_proba(x)[-1])
                    # print preds.shape
                    # print preds
                    # time.sleep(10)
                    # next_index = self.sample(preds, diversity)
                    # next_char = indices_char[next_index]

                    print "Predicted array of length %d:", self.predict(x).shape, self.predict(x)

                    next_index = self.predict(x)[-1]
                    next_char = indices_char[next_index]

                    generated += next_char
                    sentence = sentence[1:] + next_char
                    sys.stdout.write(next_char)
                    sys.stdout.flush()
                    time.sleep(1)

            self.learning_rate *= self.learning_rate_decay

        if save:
            self.save("my_theano_model.p")

    def save(self, some_path):
        logger.debug("Saving model weights to \t--------> %s" % (some_path))
        with open(some_path, 'wb') as f:
            for param in self.params:
                pickle.dump(param.get_value(borrow=True), f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, model_weights):
        logger.debug("Loading model weights from %s" % (model_weights))
        with open(model_weights, 'r') as f:
            for param in self.params:
                param.set_value(pickle.load(f))


# def test_softmax(n_u=2, n_h=6, n_y=3, time_steps=2, n_seq=3, n_epochs=1):
#     """
#     Here we test the RNN with softmax output.
#     We randomly generate `n_seq` sequences of length `time_steps`.
#     Then we make a delay and make classed which are obtained
#     using comparison to get the targets.
#     Resulting graphs are saved under the name of `softmax.png`.
#     """
#
#     # n_y is equal to the number of classes
#     print 'Testing model with softmax outputs'
#
#     np.random.seed(0)
#
#     seq = np.random.randn(n_seq, time_steps, n_u)
#     # Note that is this case `targets` is a 2d array
#     targets = np.zeros((n_seq, time_steps), dtype=np.int)
#
#     thresh = 0.5
#
#     # Comparisons to assing a class label in output
#     targets[:, 2:][seq[:, 1:-1, 1] > seq[:, :-2, 0] + thresh] = 1
#     targets[:, 2:][seq[:, 1:-1, 1] < seq[:, :-2, 0] - thresh] = 2
#     # otherwise class is 0
#
#     model = RNN(n_u=n_u, n_h=n_h, n_y=n_y,
#                 activation='lstm', output_type='softmax',
#                 learning_rate=0.001, learning_rate_decay=0.999,
#                 L1_reg=0, L2_reg=0,
#                 initial_momentum=0.5, final_momentum=0.9,
#                 momentum_switchover=5,
#                 n_epochs=n_epochs)
#
#     model.build_train(seq, targets)
#
#     logger.info("Done training the model ! ")
#
#     # plt.close('all')
#     # fig = plt.figure()
#     #
#     # ax1 = plt.subplot(311)
#     # plt.plot(seq[1])
#     # plt.grid()
#     # ax1.set_title('input')
#     # ax2 = plt.subplot(312)
#     #
#     # plt.scatter(xrange(time_steps), targets[1], marker = 'o', c = 'b')
#     # plt.grid()
#     #
#     # guess = model.predict_proba(seq[1])
#     # guessed_probs = plt.imshow(guess.T, interpolation = 'nearest', cmap = 'gray')
#     # ax2.set_title('blue points: true class, grayscale: model output (white mean class)')
#     #
#     # ax3 = plt.subplot(313)
#     # plt.plot(model.errors)
#     # plt.grid()
#     # ax3.set_title('Training error')
#     # plt.savefig('softmax_' + str(model.activation) + '_Epoch: ' + str(n_epochs) + '.png')
#

if __name__ == "__main__":
    t0 = time.time()
    # test_real()
    # test_binary()
    # test_softmax()
    print "Elapsed time: %f" % (time.time() - t0)
