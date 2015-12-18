#coding: utf-8
from __future__ import print_function, division

from argparse import ArgumentParser
import math
import sys
import time

import numpy as np

import cPickle
import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers


p = ArgumentParser()
p.add_argument('--gpu', '-G', default=-1, type=int,
               help='GPU ID (negative value indicates CPU)')
p.add_argument('-M', '--model', type=str, help='model_file')
p.add_argument('-O', '--output_file', type=str, default=sys.stderr,
               help='output file (default: stderr)')
p.add_argument('-N', '--n_lines', type=int, default=1000,
               help='num of lines')

args = p.parse_args()

if args.output_file != sys.stderr:
    args.output_file = open(args.output_file, "w")

# Prepare RNNLM model
model = cPickle.load(open(args.model, 'rb'))
vocab = cPickle.load(open("vocab.pkl", "rb"))
inv_vocab = cPickle.load(open("inv_vocab.pkl", "rb"))

n_units = model.embed.W.shape[1]

xp = cuda.cupy if args.gpu >= 0 else np

if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(args.gpu).use()
    model.to_gpu()

if args.gpu < 0:
    model.to_cpu()


def forward_one_step(x_data, state, train=True):
    if args.gpu >= 0:
        x_data = cuda.to_gpu(x_data)
    x = chainer.Variable(x_data, volatile=not train)
    h0 = model.embed(x)
    h1_in = model.l1_x(F.dropout(h0, train=train)) + model.l1_h(state['h1'])
    c1, h1 = F.lstm(state['c1'], h1_in)
    h2_in = model.l2_x(F.dropout(h1, train=train)) + model.l2_h(state['h2'])
    c2, h2 = F.lstm(state['c2'], h2_in)
    y = model.l3(F.dropout(h2, train=train))
    state = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}

    return state, F.softmax(y)


def make_initial_state(batchsize=1, train=True):
    return {name: chainer.Variable(xp.zeros((batchsize, n_units),
                                            dtype=np.float32),
                                   volatile=not train)
            for name in ('c1', 'h1', 'c2', 'h2')}


def generate_line():
    # start with <s> :1 is assigned to <s>
    index = 1
    # start with random word or particular word
    # index = vocab["パスタ"]
    # index = np.random.randint(3, len(vocab))

    cur_word = xp.array([index], dtype=xp.int32)

    state = make_initial_state(batchsize=1, train=False)
    if args.gpu >= 0:
        for key, value in state.items():
            value.data = cuda.to_gpu(value.data)

    state, predict = forward_one_step(cur_word, state, train=False)
    probability = cuda.to_cpu(predict.data)[0].astype(np.float64)
    probability /= np.sum(probability)
    index = np.random.choice(range(len(probability)), p=probability)
    cur_word = xp.array([index], dtype=xp.int32)

    seq_len = 1
    while(index != 2):  # 2 is assigned to </s>
        print(inv_vocab[index], file=args.output_file, end=" ")

        state, predict = forward_one_step(cur_word, state, train=False)
        probability = cuda.to_cpu(predict.data)[0].astype(np.float64)
        probability /= np.sum(probability)
        index_temp = np.random.choice(range(len(probability)), p=probability)
        # index_temp = cuda.to_cpu(predict.data)[0].astype(np.float64).argmax()

        if seq_len < 10 and index_temp == 1:
            seq_len += 1
            continue
        else:
            index = index_temp

        cur_word = xp.array([index], dtype=np.int32)

        seq_len += 1
        if seq_len > 30:
            break


if __name__ == '__main__':
    for i in xrange(args.n_lines):
        generate_line()
        print(file=args.output_file)
