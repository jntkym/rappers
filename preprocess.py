#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import time
import sys
import logging
import sys
import subprocess
import codecs
from utils import *


reload(sys)
sys.setdefaultencoding('utf8')
logger = logging.getLogger(__name__)

# pattern for removing redundunt spaces
space_pat=re.compile(u'\s\s+', re.U)

# # pattern for removing English - not used now
# eng_words_pat = re.compile(u'[A-Za-z]*',re.U)

# aux 1.1+
def translate_non_alphanumerics(to_translate, translate_to=None):
    not_letters_or_digits = u'!"&#%\'()*＊+,-./:;<=>?@[\]^_`{|}~...…「〜＞ｒ（）＜｀」％＿・'
    translate_table = dict((ord(char), translate_to) for char in not_letters_or_digits)
    return to_translate.translate(translate_table)

#aux 1+
def clean_lyrics(lyrics_file):
    """
    Take crawled data and do some simple cleaning on it
    :param lyrics_file: file from Otani san - crawled data
    :return: cleaned data, which will be fed to Kytea
    """
    data_corpus = []
    with open(lyrics_file) as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        for row in reader:
            sentences = row[2].decode("utf-8").strip().split(u"<BR>")
            for sentence in sentences:

                sentence = translate_non_alphanumerics(sentence)
                sentence = space_pat.sub(u' ', sentence.strip())

                # delete English
                # sentence = eng_words_pat.sub(u'', sentence).split(u"\s")

                # sentence = sentence.split(u'')
                # sentence.append(u".")
                sentence += u'.'

                data_corpus.append(sentence)

    logger.info(" Done cleaning crawled data! ")

    # saving the corpus
    with codecs.open("cleaned_lyrics.txt","w",'UTF-8') as f:
        f.write("\n".join(data_corpus))


# main func
def create_corpus(crawled_lyrics_file,save=True):
    """
    Load cleaned crawled corpus from local folder, feed to Kytea, get the output.
    Then laod the output and do final post-processing.

    :param crawled_lyrics_file:
    :param save:
    :return: clean_corpus.txt file which is fed to LSTM
    """

    # generating cleaned lyrics corpus from crawled data
    clean_lyrics(crawled_lyrics_file) # the corpus is one sequence of characters per line
    subprocess.call('kytea < cleaned_lyrics.txt > kytea_out.txt', shell=True) # processing with kytea
    logger.info(" Done kytea processing! ")

    pron=[]
    unk_pat = re.compile(u"/.*UNK")
    slash_pat = re.compile(ur"\\")

    with codecs.open("kytea_out.txt",'UTF-8') as f:
        for line in f:
            line=line.decode(encoding="utf-8").strip()
            unk_pat.sub(u"",line)
            slash_pat.sub(u"",line)
            triplets=line.split(u" ") # take a look at Kytea output: https://github.com/chezou/Mykytea-python
            seq = []
            for item in triplets:
                try:
                    hir = item.split(u"/")[2]
                    if hir != "UNK":
                        seq.append(hir)
                except IndexError:
                    continue

            if len(seq)>1:
                pron.append(u" ".join(seq))
            else:
                pron.append(u"\n")

    NN_input = u"\n".join(pron)
    logger.info(" Done generating NN input! ")

    if save:
        with open("clean_corpus.txt","w") as f:
            f.write(NN_input)
    return NN_input
