# -*- coding: utf-8 -*-
from __future__ import division
import sys
import codecs
from random import randint
from make_features import *
dummy_fill = u""
k_prev = 5
# add more features here, or modify the set of features.
ALL_FEATURES = ['LineLength', 'BOW', 'BOW5']

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
            return lines[sentence_id]

def get_all_features(history, nextLine):
    # add more features here.
    line_length = calc_linelength_score(nextLine, history[-1])
    BoW = calc_BoW_k_score(nextLine, history, k=1)
    BoW5 = calc_BoW_k_score(nextLine, history, k=k_prev)
    return {'LineLength':line_length, 'BOW':BoW, 'BOW5':BoW5}

def print_instance_features(history, nextLine, neg_num=1):
    neg_lines = []
    while len(neg_lines) < neg_num:
        randLine = get_random_line()
        if randLine == None or randLine == "":
            continue
        if randLine == nextLine:
            continue
        neg_lines.append(randLine)
    # print feature string for positive example.
    pos_features = get_all_features(history, nextLine)
    for f in ALL_FEATURES:
        feature_str = ["%s:%.4f" % (x, pos_features[x]) for x in ALL_FEATURES]
        feature_str = " ".join(feature_str)
        #print feature_str
    # print feature string for negative example.
    for neg_line in neg_lines:
        neg_features = get_all_features(history, nextLine)
        for f in ALL_FEATURES:
            feature_str = ["%s:%.4f" % (x, neg_features[x]) for x in ALL_FEATURES]
            feature_str = " ".join(feature_str)
            #print feature_str

def main():
    data_path = "data/lyrics_shonan_s27_raw.tsv"
    with open(data_path, "r") as data:
        data_size = sum([1 for _ in data])
    with codecs.open(data_path, "r", encoding="utf-8") as data:
        for i, song in enumerate(data):
            prev_lines = [dummy_fill for _ in xrange(k_prev)]
            # 1行目は飛ばす
            if i == 0:
                continue
            # 要素数が足りない時はcontinue
            temp_split = song.split("\t")
            if len(temp_split) < 2:
                sys.stderr.write("too few elements: Song-id %s" % (i))
                continue
            artist, title, text = temp_split[0], temp_split[1], temp_split[2:]
            text = u" ".join(text)
            lines = text.split(u"<BR>")
            for line in lines:
                line = line.strip()
                if line == "":
                    continue
                # calculating the features.
                print_instance_features(prev_lines, line)
                #.
                prev_lines.append(line)
                if len(prev_lines) > k_prev:
                    del prev_lines[0]

if __name__ == '__main__':
    main()

