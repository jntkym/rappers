#coding: utf-8
from __future__ import print_function, division

from argparse import ArgumentParser
import math
import sys
import time
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

import numpy as np

import cPickle
import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers

import os
import socket

p = ArgumentParser()
p.add_argument('--gpu', '-G', default=-1, type=int,
               help='GPU ID (negative value indicates CPU)')
p.add_argument('-M', '--model', type=str, help='model_file')
p.add_argument('-O', '--output_file', type=str, default=sys.stderr,
               help='output file (default: stderr)')
p.add_argument('-N', '--n_lines', type=int, default=1000,
               help='num of lines')
p.add_argument('-H', '--highquality', default=False, action="store_true")

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

# load mild yankee dict
mild_yankee_dict = utils.load_csv_to_dict("mild_dict.csv")


def force_mild(state, line, word_id):
    cur_word = xp.array([word_id], dtype=np.int32)
    state, predict = forward_one_step(cur_word, state, train=False)
    if mild_yankee_dict.has_key(unicode(inv_vocab[word_id])):
        for next_word in mild_yankee_dict[unicode(inv_vocab[word_id])].split():
            line.append(next_word)
            next_word_id = xp.array([vocab[next_word]], dtype=np.int32)
            state, predict = forward_one_step(next_word_id, state, train=False)
    return predict

    
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


def generate_line(seed=None):
    # start with <s> :1 is assigned to <s>
    line = []
    state = make_initial_state(batchsize=1, train=False)

    if args.gpu >= 0:
        for key, value in state.items():
            value.data = cuda.to_gpu(value.data)

    index = 1
    cur_word = xp.array([index], dtype=xp.int32)
    
    if seed is not None:
        for word in seed:
            state, predict = forward_one_step(cur_word, state, train=False)
            index = vocab[word]
            cur_word = xp.array([index], dtype=xp.int32)
            if index != 1:
                line.append(inv_vocab[index])            

    if index == 1:
        state, predict = forward_one_step(cur_word, state, train=False)
        probability = cuda.to_cpu(predict.data)[0].astype(np.float64)
        probability /= np.sum(probability)
        index = np.random.choice(range(len(probability)), p=probability)
        line.append(inv_vocab[index])
        
    seq_len = 1
    while(1):
        predict = force_mild(state, line, index)
        probability = cuda.to_cpu(predict.data)[0].astype(np.float64)
        if args.highquality:
            sorted_prob = sorted(probability, reverse=True)
            max_10_list = sorted_prob[:10]
            for i in xrange(len(probability)):
                if probability[i] not in max_10_list:
                    probability[i] = 0
        probability /= np.sum(probability)
        index = np.random.choice(range(len(probability)), p=probability)
        # index = cuda.to_cpu(predict.data)[0].astype(np.float64).argmax()
        
        if seq_len < 10 and index == 2:
            seq_len += 1
            continue

        if index == 2: # 2 is assigned to </s>
            break

        line.append(inv_vocab[index])

        seq_len += 1
        if seq_len > 30:
            break

    return " ".join(line)+"\n"


if __name__ == '__main__':
    requestMax = 50
    PORT = 50100
    HOST = '0.0.0.0'
    sys.stderr.write("waiting...")
    for res in socket.getaddrinfo(HOST, PORT, socket.AF_UNSPEC,
                                  socket.SOCK_STREAM, 0, socket.AI_PASSIVE):
        af, socktype, proto, canonname, sa = res
        try:
            s = socket.socket(af, socktype, proto)
        except socket.error, msg:
            s = None
            continue
        try:
            s.bind(sa)
            s.listen(requestMax)
        except socket.error, msg:
            s.close()
            s = None
            continue
        break
    if s is None:
        print('could not open socket')
        sys.exit(1)
    conn, addr = s.accept()

    print('Connected by', addr)

    # pid = os.fork()

    # if pid == 0:
        # print("child process")
        # while 1:
            # msg = raw_input("> ")
            # conn.send('%s' % msg )
            # if msg == ".":
                # break;
        # sys.exit()

    while 1:
        data = conn.recv(1024)
        seed = data.split()
        if not data:
            print('End')
            break
        elif data == "close":
            print("Client is closed")
            os.kill(pid, 9)
            break
        else:
            try:
                if "<unk>" in [inv_vocab[vocab[word]] for word in seed]:
                    conn.send("sorry, out of vocabulary ... \n")
                else:
                    for i in xrange(args.n_lines):
                        line = generate_line(seed)
                        conn.send(line)
            finally:
                conn.send("finish")

    conn.close()
    sys.exit()
