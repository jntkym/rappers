# -*- coding: utf-8 -*-
from __future__ import division
from subprocess import call, check_output
import sys
import codecs
from optparse import OptionParser
from feature_extract import *

from os import path

class NextLine(object):
    def __init__(self, all_candidate_file, history_file, model, num,
                 tmp_dir='.'):
        self.model = model
        self.candidate_file = path.join(tmp_dir, "small_candidate.txt")
        self.feature_file = path.join(tmp_dir, "small_feature.dat")
        self.predict_file = path.join(tmp_dir, "small_predict.txt")
        self.history = []

        self.preprocess(all_candidate_file, history_file, num)
        self.get_feature()
        #self.predict()
        
    def preprocess(self, all_candidate_file, history_file, num):
        # randomly extract sentences from file.
        command = "shuf -n %s %s > %s" % (num, all_candidate_file, self.candidate_file)
        call(command, shell=True)
        # process history given to form of list of length 5.
        LYRICS = codecs.open(history_file, "r", encoding="utf-8")
        for h in LYRICS.readlines():
            if len(h.split()) == 0:
                continue
            self.history.append(h)
        while len(self.history) < 5:
            self.history.insert(0, dummy_fill)
        while len(self.history) > 5:
            self.history.pop(0)

    def get_feature(self):
        FEAT = codecs.open(self.feature_file, 'w', encoding="utf-8")
        with codecs.open(self.candidate_file, "r", encoding="utf-8") as SENT:
            for i, candidate in enumerate(SENT):
                feature_dic = get_all_features(self.history, candidate)
                for f in ALL_FEATURES:
                    feature_str = ["%s:%.4f" % (ALL_FEATURES.index(x) + 1, feature_dic[x]) for x in ALL_FEATURES]
                    feature_str = " ".join(feature_str)

                FEAT.write("0 qid:0 %s\n" % (feature_str))
        
    def predict(self):
        command = "svm_classify_light %s %s %s" % (self.feature_file, self.model, self.predict_file)
        # get index of largest score.
        call(command, shell=True)
        PRED = open(self.predict_file, 'r')
        scores = [float(x.rstrip()) for x in PRED.readlines()]
        max_val = max(scores)
        max_index = scores.index(max_val)
        with codecs.open(self.candidate_file, "r", encoding="utf-8") as SENT:
            for i, candidate in enumerate(SENT):
                if i == max_index:
                    return candidate.rstrip()

if __name__ == "__main__":
    SEED = u"test_lyrics.txt"
    MD = u"/zinnia/huang/rapper/model_all"
    parser = OptionParser()
    parser.add_option("-f", "--candidate_file", action="store", dest='candidate_file')
    parser.add_option("-s", "--history", action="store", default=SEED, dest='history')
    parser.add_option("-c", "--candidate_num", action="store", type='int', default=25 ,dest='candidate_num')
    parser.add_option("-l", "--song_length", action="store", type='int', default=10 ,dest='song_length')
    parser.add_option("-m", "--model", action="store", default=MD, dest='model_path')
    options, args = parser.parse_args()
    
    for i in range(options.song_length):
        with open(SEED, 'a') as file:
            hello = NextLine(options.candidate_file, options.history, options.model_path, options.candidate_num)
            temp = hello.predict().split()
            temp.pop(0)
            temp.pop(-1)
            file.write("%s\n" % ("".join(temp)))
        file.close()

