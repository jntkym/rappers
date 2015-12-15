# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import codecs
#import yaml
import sys

sys.stdin  = codecs.getreader('UTF-8')(sys.stdin)
sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)

# http://www.asahi-net.or.jp/~ax2s-kmtn/ref/unicode/u3040.html 

class WordEmbedding:
    CHARACTERS = u"ぐだばむゐぁけちぱめゑあげぢひもをぃこっびゃんいごつぴやゔぅさづふゅゕうざてぶゆゖぇしでぷょえじとへよぉすどべら゙゙゙゙おずなぺりかせにほるがぜぬぼれきそねぽろぎぞのまゎくたはみわゟ"
    def __init__(self, word):
        self.word = word
        self._getWordEmbedding()

    def _getWordEmbedding(self):
        embedding = dict.fromkeys(list(CHARACTERS), 0)
        word_list = list(self.word)
        for i in range(len(word_list)):
            w = word_list[i]
            index = self.CHARACTERS.index(w)
            
        

if __name__ == "__main__":
    hello = WordEmbedding(u"おはよう")
