import tensorflow as tf
import numpy as np
import yaml
import sys

sys.stdin  = codecs.getreader('UTF-8')(sys.stdin)
sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)

# http://www.asahi-net.or.jp/~ax2s-kmtn/ref/unicode/u3040.html 

class WordEmbedding:
    CHARACTERS = u"ぐだばむゐぁけちぱめゑあげぢひもをぃこっびゃんいごつぴやゔぅさづふゅゕうざてぶゆゖぇしでぷょえじとへよぉすどべら゙゙゙゙おずなぺりかせにほるがぜぬぼれきそねぽろぎぞのまゎくたはみわゟ"
    def __init__(self, word):

    def _getWordEmbedding(self, ):
        embedding = dict.fromkeys(list(CHARACTERS), 0)
         
