# -*- coding: utf-8 -*-
from __future__ import division
from csv import DictReader
from os import path
import sys
import codecs

import rhyme
import utils

u"""Make features

DEMO:
python make_features.py
"""

DIR_SCRIPT = path.dirname(path.abspath(__file__))
DIR_ROOT = DIR_SCRIPT  # TODO: move this file to ./features

table_term_vowel = utils.load_csv_to_dict(path.join(DIR_ROOT,
                                                    'data/term_vowel_table.csv'))

def calc_Jaccard_similarity(BoW1, BoW2):
    # 重複を許さない場合
    # all_words = list(set(BoW1.extend(BoW2)))
    # common_words = list(set(BoW1) & set(BoW2))

    # 重複を許す場合
    assert isinstance(BoW1, list)
    assert isinstance(BoW2, list)
    all_words = BoW1+BoW2
    common_words = []
    for word in BoW1:
        if word in BoW2:
            common_words.append(word)
            BoW2.remove(word)

    return len(common_words)/len(all_words)


def calc_BoW_k_score(line, prev_lines, k=5):
    BoW_cur = line.split()
    BoW_prev = u" ".join(prev_lines[-k:]).split()
    return calc_Jaccard_similarity(BoW_cur, BoW_prev)


def calc_linelength_score(line1, line2):
    len_1 = len(line1.split())
    len_2 = len(line2.split())
    return 1 - abs(len_1 - len_2)/max(len_1, len_2)


def calc_endrhyme_score(line1, line2):
    u"""Calculate EndRhyme score

    EndRhyme is the number of matching vowel phonemes
    at the end of lines l and s_m, i.e., the last line of B.
    Spaces and consonant phonemes are ignored.

    Args: two strings (utf-8)
    """
    # Get reversed vowels
    vowels1 = rhyme.get_phonetic_transcription(line1, table_term_vowel)[::-1].replace(' ', '')
    vowels2 = rhyme.get_phonetic_transcription(line2, table_term_vowel)[::-1].replace(' ', '')

    # Count # of matching vowel phonemes
    i = 0
    n_limit = min(len(vowels1), len(vowels2))
    n_matches = 0
    while i < n_limit:
        if vowels1[i] != vowels2[i]:
            break
        n_matches += 1
        i += 1

    return n_matches


def calc_endrhyme_score_juman(line1, line2):
    u"""Calculate EndRhyme score

    EndRhyme is the number of matching vowel phonemes
    at the end of lines l and s_m, i.e., the last line of B.
    Spaces and consonant phonemes are ignored.

    Args: two strings (utf-8)
    """
    # Get reversed vowels
    vowels1 = rhyme.get_phonetic_transcription_juman(line)[::-1].replace(' ', '')
    vowels2 = rhyme.get_phonetic_transcription_juman(line2)[::-1].replace(' ', '')

    # Count # of matching vowel phonemes
    i = 0
    n_limit = min(len(vowels1), len(vowels2))
    n_matches = 0
    while i < n_limit:
        if vowels1[i] != vowels2[i]:
            break
        n_matches += 1
        i += 1

    return n_matches


# 参考までにmain関数をつけておく
def main():
    dummy_fill = u""
    k_prev = 5

    data_path = "data/lyrics_shonan_s27_raw.tsv"
    data_size = 0
    with codecs.open(data_path, "r", encoding="utf-8") as f:
        for i, row in enumerate(DictReader(f, delimiter='\t')):
            prev_lines = [dummy_fill for _ in xrange(k_prev)]

            lines = row[u"text"].split(u"<BR>")
            for line in lines:
                line = line.strip()
                if line == "":
                    continue
                line_length = calc_linelength_score(line, prev_lines[-1])
                BoW = calc_BoW_k_score(line, prev_lines, k=1)
                BoW5 = calc_BoW_k_score(line, prev_lines, k=k_prev)
                endrhyme = calc_endrhyme_score(line, prev_lines[-1])
                endrhyme_1 = calc_endrhyme_score(line, prev_lines[-2])

                datum = {"artist": row[u"artist"],
                         "title": row[u"title"],
                         "endrhyme": endrhyme,
                         "endrhyme-1": endrhyme_1,
                         "BoW": BoW,
                         "BoW5": BoW5,
                         "line_length": line_length,
                         "orig_line": line,
                         }

                prev_lines.append(line)
                if len(prev_lines) > k_prev:
                    del prev_lines[0]

                if datum["endrhyme"] >= 3:
                    print(u"{}\n<-> {}".format(line, prev_lines[-2]))

            sys.stderr.write(u"\r {} done".format(i))
            sys.stderr.flush()

if __name__ == '__main__':
    main()
