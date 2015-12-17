#!/usr/bin/env python
# -*- coding:utf-8 -*-

# from __future__ import unicode_literals
import codecs
import sys

reload(sys)
sys.setdefaultencoding('utf8')

# ranges of ordinals of unicode ideographic characters
ranges = [
    {"from": ord(u"\u3300"), "to": ord(u"\u33ff")},  # compatibility ideographs
    {"from": ord(u"\ufe30"), "to": ord(u"\ufe4f")},  # compatibility ideographs
    {"from": ord(u"\uf900"), "to": ord(u"\ufaff")},  # compatibility ideographs
    # {"from": ord(u"\U0002f800"), "to": ord(u"\U0002fa1f")}, # compatibility ideographs
    {"from": ord(u"\u30a0"), "to": ord(u"\u30ff")},  # Japanese Kana
    {"from": ord(u"\u2e80"), "to": ord(u"\u2eff")},  # cjk radicals supplement
    {"from": ord(u"\u4e00"), "to": ord(u"\u9fff")},
    {"from": ord(u"\u3400"), "to": ord(u"\u4dbf")},
    # {"from": ord(u"\U00020000"), "to": ord(u"\U0002a6df")},
    # {"from": ord(u"\U0002a700"), "to": ord(u"\U0002b73f")},
    # {"from": ord(u"\U0002b740"), "to": ord(u"\U0002b81f")},
    # {"from": ord(u"\U0002b820"), "to": ord(u"\U0002ceaf")}  # included as of Unicode 8.0
]


def is_cjk(char):
    return any([range["from"] <= ord(char) <= range["to"] for range in ranges])


def load_csv_to_dict(filepath, delimiter=','):
    u"""Load a .csv file to dictionary

    Args:
    - filepath: path to the csv file
    - delimiter: delimiter in the csv file

    Input: csv file
      key,val
      ...

    Return:
      dictionary
    """
    table = {}
    with codecs.open(filepath,
                     'r', encoding='utf-8') as f:
        for line in f:
            key, val = line.strip().split(delimiter)
            table[key] = val
    return table
