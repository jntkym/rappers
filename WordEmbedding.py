# -*- coding: utf-8 -*-
import numpy as np
import codecs
import sys

sys.stdin  = codecs.getreader('UTF-8')(sys.stdin)
sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)

# http://www.asahi-net.or.jp/~ax2s-kmtn/ref/unicode/u3040.html 

class WordEmbedding:
    PADDING = u"％"
    CHARACTERS = u"ぐだばむゐぁけちぱめゑあげぢひもをぃこっびゃんいごつぴやゔぅさづふゅゕうざてぶゆゖぇしでぷょえじとへよぉすどべらおずなぺりかせにほるがぜぬぼれきそねぽろぎぞのまゎくたはみわゟＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
    VOCAB = PADDING + CHARACTERS
    EMBEDDING_SIZE = len(VOCAB)*3
    ALPHA = 0.2
    def __init__(self, word):
        self.word = word
        self.embedding = dict.fromkeys(list(self.VOCAB), 0)
        self.inv_embedding = dict.fromkeys(list(self.VOCAB), 0)
        self.frequency = dict.fromkeys(list(self.VOCAB), 0)
        self._getWordEmbeddings()
        self.vector = self.embedding.values() + self.inv_embedding.values() + self.frequency.values()

    def _getWordEmbeddings(self):
        word_list = list(self.word)
        wordlen = normalizer = len(word_list)
        for position in range(len(word_list)):
            w = word_list[position]
            self.frequency[w] += 1

            index = self.VOCAB.index(w)
            position_weight = ((1 - self.ALPHA) ** position) / normalizer
            self.embedding[w] += position_weight
            inv_position_weight = ((1 - self.ALPHA) ** (wordlen - position - 1)) / normalizer
            self.inv_embedding[w] += inv_position_weight


if __name__ == "__main__":
    hello = WordEmbedding(u"おはよう")
    #print hello.vector

