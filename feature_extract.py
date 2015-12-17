# -*- coding: utf-8 -*-
from __future__ import division
import sys
import codecs
import getpass
from optparse import OptionParser
from random import randint
from make_features import *
dummy_fill = u""
k_prev = 5
# add more features here, or modify the set of features.
SCRIPT_PREFIX = "nice -n 19 python feature_extract.py --song_id %s --qid %s > /data/%s/rapper/features/%s.dat" % ('%s', '%s', getpass.getuser(), '%s')
ALL_FEATURES = ['LineLength', 'BOW', 'BOW5', 'EndRhyme', 'EndRhyme-1']

def get_random_line():
    data_path = "data/lyrics_shonan_s27_raw.tsv"
    with open(data_path, "r") as data:
        data_size = sum([1 for _ in data])
    # randomly choose a song.
    song_id = randint(1, data_size)
    with codecs.open(data_path, "r", encoding="utf-8") as data:
        for i, song in enumerate(data):
            if i != song_id:
                continue
            temp_split = song.split("\t")
            # 要素数が足りない時はreturn None.
            if len(temp_split) < 2:
                return None
            artist, title, text = temp_split[0], temp_split[1], temp_split[2:]
            text = u" ".join(text)
            lines = text.split(u"<BR>")
            song_size = len(lines)
            sentence_id = randint(0, song_size - 1)
            # randomly choose a sentence.
            if len(lines[sentence_id].split()) != 0:
                return lines[sentence_id].rstrip()
            elif u"くり返し" in lines[sentence_id]:
                return None
            else :
                return None

def get_all_features(history, nextLine):
    # add more features here.
    line_length = calc_linelength_score(nextLine, history[-1])
    BoW = calc_BoW_k_score(nextLine, history, k=1)
    BoW5 = calc_BoW_k_score(nextLine, history, k=k_prev)
    endrhyme = calc_endrhyme_score(nextLine, history[-1])
    endrhyme_1 = calc_endrhyme_score(nextLine, history[-2])
    return {'LineLength' : line_length, 'BOW' : BoW, 'BOW5' : BoW5, \
            'EndRhyme' : endrhyme, 'EndRhyme-1' : endrhyme_1}

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
    data_path = "data/lyrics_shonan_s27_raw.tsv"
    qid_now = start_qid
    with codecs.open(data_path, "r", encoding="utf-8") as data:
        for i, song in enumerate(data):
            if i != song_id:
                continue
            # start processing song.
            prev_lines = [dummy_fill for _ in xrange(k_prev)]
            temp_split = song.split("\t")
            artist, title, text = temp_split[0], temp_split[1], temp_split[2:]
            text = u" ".join(text)
            lines = text.split(u"<BR>")
            for line in lines:
                line = line.strip()
                if line == "":
                    continue
                print_instance_features(qid_now, prev_lines, line)
                # maintainance.
                qid_now += 1
                prev_lines.append(line)
                if len(prev_lines) > k_prev:
                    del prev_lines[0]

def generate_task():
    data_path = "data/lyrics_shonan_s27_raw.tsv"
    qid_now = 0
    with codecs.open(data_path, "r", encoding="utf-8") as data:
        for i, song in enumerate(data):
            # 1行目は飛ばす
            if i == 0:
                continue
            # 要素数が足りない時はcontinue
            temp_split = song.split("\t")
            if len(temp_split) < 2:
                sys.stderr.write("too few elements: Song-id %s\n" % (i))
                continue
            # print task.
            print SCRIPT_PREFIX % (i, qid_now, i)
            #
            artist, title, text = temp_split[0], temp_split[1], temp_split[2:]
            text = u" ".join(text)
            lines = text.split(u"<BR>")
            for line in lines:
                line = line.strip()
                if line == "":
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

    print_song_features(options.song_id, options.qid)

