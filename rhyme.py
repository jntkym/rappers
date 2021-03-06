#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import argparse
from csv import DictReader
import codecs
import logging
from os import path
import re
import sys

from pyknp import Juman

import utils
import preprocess

reload(sys)
sys.setdefaultencoding('utf8')

verbose = False
logger = None

DIR_SCRIPT = path.dirname(path.abspath(__file__))
DIR_ROOT = DIR_SCRIPT  # TODO: move this file to ./features

PATH_KANA_VOWEL_TABLE = path.join(DIR_ROOT,
                                  'data/kana_vowel_table.csv')
PATH_EN_KANA_TABLE = path.join(DIR_ROOT,
                               'data/en_kana_table.csv')

def init_logger():
    global logger
    logger = logging.getLogger('Rhyme')
    logger.setLevel(logging.WARNING)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)

def get_phonetic_transcription(text, table_term_vowels):
    result = []
    for chunk in text.split():
        try:
            result.append(table_term_vowels[chunk])
        except KeyError:
            pass
    return ' '.join(result)


def get_phonetic_transcription_juman(text):
    u"""Return vowels using juman (slow)
    """
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
    table_term_vowel = utils.load_csv_to_dict(path.join(DIR_ROOT,
                                                        'data/term_vowel_table.csv'))
    with codecs.open(args.filename, encoding='utf-8') as f:
        for line in f:
            i += 1
            if verbose:
                if i%10 == 0:
                    logger.info(i)
            if len(line) == 0:
                continue
            result1 = get_phonetic_transcription(line.decode('utf-8'),
                                                table_term_vowel)
            result2 = get_phonetic_transcription_juman(line.decode('utf-8'))
            print('{}'.format(line.strip()))
            print(''.join(result1.split()))
            print(result2)
            print('')
    return 0


if __name__ == '__main__':
    init_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
