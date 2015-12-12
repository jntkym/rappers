#!/usr/bin/env python
# -*- coding:utf-8 -*-

# from __future__ import unicode_literals
import numpy as np
import re
import time

class Vocab:
    __slots__ = ["word2index", "index2word", "unknown"]

    def __init__(self, index2word = None):
        self.word2index = {}
        self.index2word = []

        # add unknown word:
        self.add_words(["**UNKNOWN**"])
        self.unknown = 0

        if index2word is not None:
            self.add_words(index2word)

    def add_words(self, words):
        for word in words:
            if word not in self.word2index:
                self.word2index[word] = len(self.word2index)
                self.index2word.append(word)

    def __call__(self, line):
        """
        Convert from numerical representation to words
        and vice-versa.
        """
        if type(line) is np.ndarray:
            return " ".join([self.index2word[word] for word in line])
        if type(line) is list:
            if len(line) > 0:
                if line[0] is int:
                    return " ".join([self.index2word[word] for word in line])
            indices = np.zeros(len(line), dtype=np.int32)
        else:
            line = line.split(" ")
            indices = np.zeros(len(line), dtype=np.int32)

        for i, word in enumerate(line):
            indices[i] = self.word2index.get(word, self.unknown)

        return indices

    @property
    def size(self):
        return len(self.index2word)

    def __len__(self):
        return len(self.index2word)


ranges = [
  {"from": ord(u"\u3300"), "to": ord(u"\u33ff")},         # compatibility ideographs
  {"from": ord(u"\ufe30"), "to": ord(u"\ufe4f")},         # compatibility ideographs
  {"from": ord(u"\uf900"), "to": ord(u"\ufaff")},         # compatibility ideographs
  # {"from": ord(u"\U0002f800"), "to": ord(u"\U0002fa1f")}, # compatibility ideographs
  {"from": ord(u"\u30a0"), "to": ord(u"\u30ff")},         # Japanese Kana
  {"from": ord(u"\u2e80"), "to": ord(u"\u2eff")},         # cjk radicals supplement
  {"from": ord(u"\u4e00"), "to": ord(u"\u9fff")},
  {"from": ord(u"\u3400"), "to": ord(u"\u4dbf")},
  # {"from": ord(u"\U00020000"), "to": ord(u"\U0002a6df")},
  # {"from": ord(u"\U0002a700"), "to": ord(u"\U0002b73f")},
  # {"from": ord(u"\U0002b740"), "to": ord(u"\U0002b81f")},
  # {"from": ord(u"\U0002b820"), "to": ord(u"\U0002ceaf")}  # included as of Unicode 8.0
]

def is_cjk(char):
    return any([range["from"] <= ord(char) <= range["to"] for range in ranges])

def get_cjk_characters(cjk_string):
    p = re.compile( '[A-Za-z]*]')
    # i = 0
    # character_list=[]
    # while i<len(cjk_string):
    #     if is_cjk(cjk_string[i]):
    #         character_list.append(cjk_string[i])
    #     else:
    #         pass
    #     i += 1
    # return character_list
    p.sub(' ', cjk_string)
    # x = re.sub(r"\s+([a-zA-Z_][a-zA-Z_0-9]*)",r" ", cjk_string)
    # return x

# def get_letter_seq(string):
#     words = []
#     seq=""
#     letter_flag=1
#
#     print string
#
#     for ch in string:
#         while letter_flag:
#             if ch.isalpha():
#                 seq += ch
#             elif is_cjk(ch):
#                 words.append(seq)
#                 seq = ""
#
#     print "Words:", words
#     return words
#     # pattern = re.search(r"(.*)([^A-Za-z])(.*)", string)
#     # word1 = " ".join(re.findall(".*[a-zA-Z]", s))
#     # print word1
#     #
#     # if len(word1) > 0:
#     #     words.append(word1)
#
#     return words

