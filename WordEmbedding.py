# -*- coding: utf-8 -*-
import numpy as np
import codecs
import sys
ALPHA = 0.2

sys.stdin  = codecs.getreader('UTF-8')(sys.stdin)
sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)

# http://www.asahi-net.or.jp/~ax2s-kmtn/ref/unicode/u3040.html 

class WordEmbedding:
    CHARACTERS = u"ぐだばむゐぁけちぱめゑあげぢひもをぃこっびゃんいごつぴやゔぅさづふゅゕうざてぶゆゖぇしでぷょえじとへよぉすどべら゙゙゙゙おずなぺりかせにほるがぜぬぼれきそねぽろぎぞのまゎくたはみわゟＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
    SIZE = len(CHARACTERS)*3
    def __init__(self, word):
        self.word = word
        self.embedding = dict.fromkeys(list(self.CHARACTERS), 0)
        self.inv_embedding = dict.fromkeys(list(self.CHARACTERS), 0)
        self.frequency = dict.fromkeys(list(self.CHARACTERS), 0)
        self._getWordEmbeddings()
        self.vector = np.array(self.embedding.values() + self.inv_embedding.values() + self.frequency.values())

    def _getWordEmbeddings(self):
        word_list = list(self.word)
        Z = len(word_list)
        for position in range(len(word_list)):
            w = word_list[position]
            self.frequency[w] += 1

            index = self.CHARACTERS.index(w)
            position_weight = ((1 - ALPHA) ** position) / Z
            self.embedding[w] += position_weight
            inv_position_weight = ((1 - ALPHA) ** (Z - position - 1)) / Z
            self.inv_embedding[w] += inv_position_weight


if __name__ == "__main__":
    hello = WordEmbedding(u"おはよう")
    #print hello.vector

