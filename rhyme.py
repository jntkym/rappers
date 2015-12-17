#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import argparse
from csv import DictReader
import codecs
import logging
import re
import sys

from pyknp import Juman

import preprocess


reload(sys)
sys.setdefaultencoding('utf8')

verbose = False
logger = None

PATH_KANA_VOWEL_TABLE = './data/kana_vowel_table.csv'
PATH_EN_KANA_TABLE = './data/en_kana_table.csv'

def init_logger():
    global logger
    logger = logging.getLogger('Logger')  # to be changed
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)

def get_phonetic_transcription(text, output_fname=None):
    r = re.compile(u'\w+')
    juman = Juman()
    # (Hira+Kata)Kana -> vowel
    kana_vowel_table = {}
    with codecs.open(PATH_KANA_VOWEL_TABLE,
                     'r', encoding='utf-8') as f:
        for line in f:
            kana, vowel = line.strip().split(',')
            kana_vowel_table[kana] = vowel

    # English -> Japanese Hiragana
    en_kana_table = {}
    with codecs.open(PATH_EN_KANA_TABLE,
                     'r', encoding='utf-8') as f:
        for line in f:
            eng, kana = line.strip().split(',')
            en_kana_table[eng] = kana

    result = []
    text = preprocess.translate_non_alphanumerics(text)
    for chunk in text.split():
        vowels = []
        mrphs = juman.analysis(chunk)
        for mrph in mrphs:
            yomi = mrph.yomi.lower()
            if yomi in en_kana_table:
                yomi = en_kana_table[yomi]
            m = r.search(yomi)
            if m is not None:
                # use stderr to print if needed.
                #sys.stderr.write(m.group(0))
                pass
            for idx in range(len(yomi)):
                kana = yomi[idx]
                if kana == u'ー':
                    try:
                        vowels.append(vowels[-1])
                    except KeyError:
                        logger.info(chunk)
                    continue
                if kana in [u'ゃ', u'ャ']:
                    try:
                        vowels[-1] = 'a'
                    except KeyError:
                        logger.info(chunk)
                    continue
                if kana in [u'ゅ', u'ュ']:
                    try:
                        vowels[-1] = 'u'
                    except KeyError:
                        logger.info(chunk)
                    continue
                if kana in [u'ょ', u'ョ']:
                    try:
                        vowels[-1] = 'o'
                    except KeyError:
                        logger.info(chunk)
                    continue
                try:
                    vowels.append(kana_vowel_table[kana])
                except KeyError:
                    pass
                    # logger.info(kana)
        if len(vowels) > 0:
            result.append(''.join(vowels))
    return ' '.join(result)


def main(args):
    global verbose
    verbose = args.verbose

    i = 0
    with codecs.open(args.filename, 'r', encoding='utf-8') as f:
        for row in DictReader(f, delimiter='\t'):
            i += 1
            if verbose:
                if i%10 == 0: logger.info(i)
            lines = row['text'].split('<BR>')
            for line in lines:
                if len(line) == 0:
                    continue
                result = get_phonetic_transcription(line.decode('utf-8'))
    return 0


if __name__ == '__main__':
    init_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
