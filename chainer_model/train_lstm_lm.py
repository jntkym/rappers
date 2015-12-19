#!/usr/bin/env python

from __future__ import print_function
import argparse
import math
import sys
import os
import time
import copy

from collections import defaultdict
import numpy as np
import six

import cPickle
import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-G', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--n_epoch', '-I', default=100, type=int,
                    help='num_epoch')
parser.add_argument('--batchsize', '-B', default=100, type=int,
                    help='batchsize')
parser.add_argument('--vocabsize', '-V', default=100000, type=int,  # true vocab size < 100000
                    help='vocabsize')
parser.add_argument('--corpus', '-C', default=".", type=str,  # true vocab size < 100000
                    help='training corpus')
parser.add_argument('--model', '-M', default="./models", type=str,  # true vocab size < 100000
                    help='directory in which model is saved')

args = parser.parse_args()
xp = cuda.cupy if args.gpu >= 0 else np

if not os.path.exists(args.model):
    os.makedirs(args.model)

n_units = 200  # number of units per layer
bprop_len = 20   # length of truncated BPTT
grad_clip = 5    # gradient norm threshold to clip


def make_vocab(filename, vocab_size):
    word_freq = defaultdict(lambda: 0)
    vocab = defaultdict(int)
    inv_vocab = defaultdict(int)
    num_lines = 0
    num_words = 0
    with open(filename) as fp:
        for line in fp:
            words = line.split()
            num_lines += 1
            num_words += len(words)
            for word in words:
                word_freq[word] += 1

    # 0: unk
    # 1: <s>
    # 2: </s>
    vocab['<unk>'] = 0
    inv_vocab[0] = '<unk>'
    vocab['<s>'] = 1
    inv_vocab[1] = '<s>'
    vocab['</s>'] = 2
    inv_vocab[2] = '</s>'
    i = 2
    for k, v in sorted(word_freq.items(), key=lambda x: -x[1]):
        i += 1
        if i > vocab_size:
            break
        vocab[k] = i
        inv_vocab[i] = k

    return vocab, inv_vocab, num_lines, num_words


def generate_batch(filename, batch_size):
    with open(filename) as fp:
        batch = []
        try:
            while True:
                for i in range(batch_size):
                    batch.append(next(fp).strip().split())

                max_len = max(len(x) for x in batch)
                batch = [['<s>'] + x + ['</s>'] * (max_len - len(x) + 1) for x in batch]
                yield batch

                batch = []
        except:
            pass

        if batch:
            max_len = max(len(x) for x in batch)
            batch = [['<s>'] + x + ['</s>'] * (max_len - len(x) + 1) for x in batch]
            yield batch

vocab, inv_vocab, num_lines, num_words = make_vocab(args.corpus, args.vocabsize)
cPickle.dump(vocab, open("vocab.pkl", "wb"))
cPickle.dump(inv_vocab, open("inv_vocab.pkl", "wb"))

print('#vocab =', len(vocab))

# Prepare RNNLM model
model = chainer.FunctionSet(embed=F.EmbedID(len(vocab), n_units),
                            l1_x=F.Linear(n_units, 4 * n_units),
                            l1_h=F.Linear(n_units, 4 * n_units),
                            l2_x=F.Linear(n_units, 4 * n_units),
                            l2_h=F.Linear(n_units, 4 * n_units),
                            l3=F.Linear(n_units, len(vocab)))

for param in model.parameters:
    param[:] = np.random.uniform(-0.1, 0.1, param.shape)
if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(args.gpu).use()
    model.to_gpu()


def forward_one_step(x_data, y_data, state, train=True):
    # Neural net architecture
    x = chainer.Variable(x_data, volatile=not train)
    t = chainer.Variable(y_data, volatile=not train)
    h0 = model.embed(x)
    h1_in = model.l1_x(F.dropout(h0, train=train)) + model.l1_h(state['h1'])
    c1, h1 = F.lstm(state['c1'], h1_in)
    h2_in = model.l2_x(F.dropout(h1, train=train)) + model.l2_h(state['h2'])
    c2, h2 = F.lstm(state['c2'], h2_in)
    y = model.l3(F.dropout(h2, train=train))
    state = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}

    return state, F.softmax_cross_entropy(y, t)


def make_initial_state(batchsize=args.batchsize, train=True):
    return {name: chainer.Variable(xp.zeros((batchsize, n_units),
                                            dtype=np.float32),
                                   volatile=not train)
            for name in ('c1', 'h1', 'c2', 'h2')}

# Setup optimizer
optimizer = optimizers.AdaDelta()
optimizer.setup(model)


def main():
    state = make_initial_state()
    accum_loss = chainer.Variable(xp.zeros((), dtype=np.float32))
    for epoch in range(args.n_epoch):
        print('epoch %d/%d: ' % (epoch + 1, args.n_epoch))
        log_ppl = 0.0
        trained = 0

        opt = optimizers.AdaDelta()
        opt.setup(model)

        for batch in generate_batch(args.corpus, args.batchsize):
            batch = [[vocab[x] for x in words] for words in batch]
            K = len(batch)
            if K != args.batchsize:
                break
            L = len(batch[0]) - 1

            opt.zero_grads()

            for l in range(L):
                x_batch = xp.array([batch[k][l] for k in range(K)], dtype=np.int32)
                y_batch = xp.array([batch[k][l + 1] for k in range(K)], dtype=np.int32)
                state, loss_i = forward_one_step(x_batch, y_batch, state)
                accum_loss += loss_i
                accum_loss.backward()

                log_ppl += accum_loss.data.reshape(()) * K

            accum_loss.unchain_backward()  # truncate
            accum_loss = chainer.Variable(xp.zeros((), dtype=np.float32))

            optimizer.clip_grads(grad_clip)
            optimizer.update()

            trained += K
            sys.stderr.write('\r  %d/%d' % (trained, num_lines))
            sys.stderr.flush()

        log_ppl /= float(num_words)
        print('  log(PPL) = %.10f' % log_ppl)
        # print('  PPL      = %.10f' % math.exp(log_ppl))

        if (epoch + 1) % 5 == 0:
            print("save model")
            model_name = "%s/kokkai_lstm_lm.epoch%d" % (args.model, epoch+1)
            cPickle.dump(copy.deepcopy(model).to_cpu(), open(model_name, 'wb'))

    print('training finished.')


if __name__ == '__main__':
    main()
