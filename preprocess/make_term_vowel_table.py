#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import argparse
import codecs
import logging
import re
from os import path

u"""Make Term-vowel table

"Term" includes any separated words in Japanese or English by Juman

NOTE:
This script writes results to the standard output

DEMO:
python make_term_vowel_table.py output_test.csv
"""

DIR_SCRIPT = path.dirname(path.abspath(__file__))
DIR_ROOT = path.dirname(DIR_SCRIPT)

verbose = False
logger = None


def init_logger():
    global logger
    logger = logging.getLogger('Table')
    logger.setLevel(logging.WARNING)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)

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


def main(args):
    global verbose
    verbose = args.verbose

    # Load files
    f_string = codecs.open(args.string_corpus, "r", encoding="utf-8")
    f_hiragana = codecs.open(args.hiragana_corpus, "r", encoding="utf-8")

    # Load tables
    table_kana_vowel = load_csv_to_dict(args.table_kana_vowel)
    table_en_kana = load_csv_to_dict(args.table_en_kana)

    # Term-Vowel table
    vowels = {}

    r = re.compile(u'\w+')  # This is to check if term is a English word

    unknown_terms = set()
    n_lines = 1
    line_string = f_string.readline()
    line_hiragana = f_hiragana.readline()
    while line_string and line_hiragana:
        line_string = f_string.readline()
        line_hiragana = f_hiragana.readline()
        row_string = line_string.strip().split()
        row_hiragana = line_hiragana.strip().split()

        # have the same number of terms?
        if len(row_string) != len(row_hiragana):
            logger.warning('Inconsistent number of terms')
            logger.warning(line_string)
            logger.warning(line_hiragana)

        for i in range(len(row_string)):
            term = row_string[i]
            kana = row_hiragana[i]

            # if term in vowels.keys():
            #     continue

            m = r.match(term)

            # English terms
            if m:
                try:
                    kana = table_en_kana[m.group(0)]
                except KeyError:
                    unknown_terms.add(term)
                    vowels[term] = kana
                    logger.debug('No corresponding kana:\t' + term)

            # Get vowels
            vowels[term] = ''
            for c in kana:
                if c == u'っ': continue
                if c == u'ー':
                    if len(vowels[term]) > 0:
                        vowels[term] += vowels[term][-1]
                    continue

                # If no corresponding vowel, ignore
                try:
                    vowels[term] += table_kana_vowel[c]
                except KeyError:
                    unknown_terms.add(term)
                    logger.debug('No corresponding vowel:\t' + c)
        if verbose:
            if n_lines%(10**4) == 0: logger.info('Read {} lines'.format(n_lines))
        n_lines += 1

    f_string.close()
    f_hiragana.close()

    if verbose:
        logger.info('Done: {} lines'.format(n_lines))

    # Output to standard output
    for key, val in sorted(vowels.items(), key=lambda t: t[0]):
        print('{},{}'.format(key.encode('utf-8'), val.encode('utf-8')))

    if verbose:
        logger.info('Unknown terms: {}'.format(len(unknown_terms)))
    # Write terms that did not appear at the tables (optional)
    if args.f_unknown_terms:
        with open(args.f_unknown_terms, 'w') as f:
            for term in sorted(list(unknown_terms)):
                f.write(term.encode('utf-8') + '\n')

    return 0


if __name__ == '__main__':
    init_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--string-corpus', dest='string_corpus',
                        default=path.join(DIR_ROOT, 'data/string_corpus.txt'),
                        help='path to hiragana corpus')
    parser.add_argument('--hiragana-corpus', dest='hiragana_corpus',
                        default=path.join(DIR_ROOT, 'data/hiragana_corpus.txt'),
                        help='path to hiragana corpus')
    parser.add_argument('--kana-vowel', dest='table_kana_vowel',
                        default=path.join(DIR_ROOT, 'data/kana_vowel_table.csv'),
                        help='path to kana to vowel table')
    parser.add_argument('--en-kana', dest='table_en_kana',
                        default=path.join(DIR_ROOT, 'data/en_kana_table.csv'),
                        help='path to english to kana table')
    parser.add_argument('--unknown-terms', dest='f_unknown_terms',
                        default=None,
                        help='path to file to save unknown terms')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
