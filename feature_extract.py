# -*- coding: utf-8 -*-
from __future__ import division
import sys
import re
import codecs
import getpass
import cdb
import socket
from optparse import OptionParser
from random import randint
from make_features import *
dummy_fill = u""
k_prev = 5
# add more features here, or modify the set of features.
SCRIPT_PREFIX = "nice -n 19 python feature_extract.py --song_id %s --qid %s > /data/%s/rapper/features/%s.dat" % ('%s', '%s', getpass.getuser(), '%s')
ALL_FEATURES = ['LineLength', 'BOW', 'BOW5', 'EndRhyme', 'EndRhyme-1', 'NN3']

def get_random_line():
    data_path = "data/string_corpus.cdb"
    data = cdb.init(data_path)
    # randomly choose a song.
    data_size = len(data)
    song_id = randint(1, data_size)
    thisSong = data.get("%s" % (song_id))
    thisSong = thisSong.split('\n')
    # randomly choose a sentence.
    song_size = len(thisSong)
    sentence_id = randint(0, song_size - 1)
    thisSentence = thisSong[sentence_id]

    if u"くり返し" in thisSentence:
        return None
    if len(thisSentence.split()) != 0:
        return thisSentence
    else :
        return None

def pad_line(line):
    words = line.split()
    if len(words) > 13:
        words = words[:13]
    if len(words) < 13:
        pad = ['ｐ']*(13-len(words))
        words = pad + words
    return ":".join(words)

def get_NN3_feature(nextLine, history):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except:
        sys.stderr.write('Failed to create socket.\n')
        return 0

    #host = '10.228.146.188'
    host = '10.228.146.43'
    port = 12345
    try:
        remote_ip = socket.gethostbyname( host )
    except socket.gaierror:
        sys.stderr.write('Could not resolve hostname.\n')
        return 0
    s.connect((remote_ip , port))
    # generate message.
    testData = history + [nextLine]
    testData = map(pad_line, testData)
    testData = "||".join(testData)
    # send request.
    try:
        s.sendall(testData)
    except socket.error:
        sys.stderr.write('Send failed.\n')
        return 0
    #sys.stderr.write('msg sended.\n')
    # mark the end of message.
    endMsg = "<END>"
    try:
        s.sendall(endMsg)
    except socket.error:
        sys.stderr.write('Send failed.\n')
        return 0
    #sys.stderr.write('msg ended.\n')
    # receive result.
    reply = s.recv(1024)
    #sys.stderr.write('get reply:%s\n' % (reply))
    return int(reply)

def get_all_features(history, nextLine):
    # add more features here.
    line_length = calc_linelength_score(nextLine, history[-1])
    BoW = calc_BoW_k_score(nextLine, history, k=1)
    BoW5 = calc_BoW_k_score(nextLine, history, k=k_prev)
    endrhyme = calc_endrhyme_score(nextLine, history[-1])
    endrhyme_1 = calc_endrhyme_score(nextLine, history[-2])
    NN3 = get_NN3_feature(nextLine, history[-3:])
    return {'LineLength' : line_length, 'BOW' : BoW, 'BOW5' : BoW5, \
            'EndRhyme' : endrhyme, 'EndRhyme-1' : endrhyme_1, 'NN3' : NN3}

def print_instance_features(qid, history, nextLine, neg_num=1):
    # randomly sample nextline as negative examples.
    neg_lines = []
    while len(neg_lines) < neg_num:
        randLine = get_random_line()
        if randLine == None:
            continue
        if randLine == nextLine:
            continue
        neg_lines.append(randLine)
    # print feature string for positive example.
    target = 1
    pos_features = get_all_features(history, nextLine)
    for f in ALL_FEATURES:
        feature_str = ["%s:%.4f" % (ALL_FEATURES.index(x) + 1, pos_features[x]) for x in ALL_FEATURES]
        feature_str = " ".join(feature_str)
    print "%s qid:%s %s" % (target, qid, feature_str)
    # print feature string for negative example.
    target = 0
    for neg_line in neg_lines:
        neg_features = get_all_features(history, neg_line)
        for f in ALL_FEATURES:
            feature_str = ["%s:%.4f" % (ALL_FEATURES.index(x) + 1, neg_features[x]) for x in ALL_FEATURES]
            feature_str = " ".join(feature_str)
        print "%s qid:%s %s" % (target, qid, feature_str)

def print_song_features(song_id, start_qid):
    data_path = "data/string_corpus.cdb"
    data = cdb.init(data_path)
    qid_now = start_qid
    thisSong = data.get("%s" % (song_id))
    # start processing song.
    prev_lines = [dummy_fill for _ in xrange(k_prev)]
    for line in thisSong.split('\n'):
        line = line.rstrip()
        if line == "":
            continue
        print_instance_features(qid_now, prev_lines, line)
        # maintainance.
        qid_now += 1
        prev_lines.append(line)
        if len(prev_lines) > k_prev:
            del prev_lines[0]

def generate_task():
    data_path = "data/string_corpus.cdb"
    data = cdb.init(data_path)
    data_size = len(data)
    qid_now = 0
    for i in range(data_size):
        # print task.
        print SCRIPT_PREFIX % (i, qid_now, i)
        thisSong = data.get("%s" % (i))
        for line in thisSong.split('\n'):
            if line.rstrip() =="":
                continue
            qid_now += 1
    sys.stderr.write("Total number of instance: %s\n" % (qid_now))

if __name__ == '__main__':
    sys.stdin = codecs.getreader('utf-8')(sys.stdin)
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr)

    parser = OptionParser()
    parser.add_option("-s", "--song_id", action="store", type="int", dest='song_id')
    parser.add_option("-q", "--qid", action="store", type="int", dest='qid')
    options, args = parser.parse_args()

    #generate_task()
    print_song_features(options.song_id, options.qid)
