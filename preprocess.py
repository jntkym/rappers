#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import csv
import time
import sys
import subprocess
import codecs
from utils import *
import unicodedata
import argparse
import cPickle as pickle

reload(sys)
sys.setdefaultencoding('utf8')

# pattern for removing redundunt spaces
space_pat = re.compile(u'\s\s+', re.U)

# # pattern for removing English - not used now
# eng_words_pat = re.compile(u'[A-Za-z]*',re.U)

# aux 1.1+
def translate_non_alphanumerics(to_translate, translate_to=None):
    """
    Deleting not needed symbols
    """
    not_letters_or_digits = u'[!&#%\"\'()_`{※+,』\|}~?...…「〜＞ｒ（）＜｀！」？＿％・@＠”’"：；＋ー！。。。、＿・_ _『 □**＊-\.\/:;<=>△?@\[\]\^'
    translate_table = dict((ord(char), translate_to) for char in not_letters_or_digits)
    return to_translate.translate(translate_table)


# aux 1+
def clean_lyrics(lyrics_file):
    """
    Take crawled data and do some simple cleaning on it
    :param lyrics_file: crawled data
    :return: cleaned data, which will be fed to Kytea
    """
    data_corpus = []
    with open(lyrics_file) as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        for row in reader:
            sentences = row[2].strip().split(u"<BR>")
            for sentence in sentences:
                sentence = unicode(sentence)
                sentence = translate_non_alphanumerics(sentence)
                sentence = space_pat.sub(u' ', sentence)

                if len(sentence) > 1:
                    data_corpus.append(sentence)
            data_corpus.append(u"\n")

    print(" Done cleaning crawled data! ")

    # saving the corpus
    with codecs.open("data/cleaned_lyrics.txt", "w", 'UTF-8') as f:
        f.write("\n".join(data_corpus))


# aux 2
def create_corpus(crawled_lyrics_file, save=False):
    """
    Load cleaned crawled corpus from local folder, feed to Kytea, get the output.
    Then laod the output and do post-processing.

    :param crawled_lyrics_file:
    :param save:
    :return: clean_corpus file 
    """

    # generating cleaned lyrics corpus from crawled data
    clean_lyrics(crawled_lyrics_file)  # the corpus is one sequence of characters per line
    subprocess.call('kytea < ./data/cleaned_lyrics.txt > ./data/kytea_out.txt', shell=True)  # processing with kytea
    print(" Done kytea processing! ")

    pron = []
    unk_pat = re.compile(u"/補助記号/UNK")
    slash_pat = re.compile(ur"\\")

    with codecs.open("data/kytea_out.txt", 'UTF-8') as f:
        for line in f:
            line = line.decode(encoding="utf-8")
            if line[0] == "\n":
                pron.append(u"\n")

            line = line.strip()
            line = unk_pat.sub(u"", line)
            line = slash_pat.sub(u"", line)

            triplets = line.split(u" ")  # take a look at Kytea output: https://github.com/chezou/Mykytea-python
            seq = []
            for item in triplets:
                try:
                    hir = item.split(u"/")[0]
                    if hir != "\\":
                        seq.append(hir)
                except IndexError:
                    continue

            candidate_line = unicodedata.normalize("NFKC", u" ".join(seq))
            candidate_line = re.sub(u"[A-Za-z]", u"", candidate_line)
            candidate_line = re.sub(u"\s+", u"", candidate_line)
            candidate_line = re.sub(u"\d+", u"5", candidate_line)

            if len(candidate_line) > 2:
                pron.append(candidate_line)

    juman_input = u"\n".join(pron)
    juman_input = re.sub(u"\n{4}",u"\n\n",juman_input)
    return juman_input


# main function - creates input for the NN
def clean_corpus(crawled_lyrics_file, savepath=None):

    print("Preparing data ...")
    text = create_corpus(crawled_lyrics_file, save=False).lower()

    if savepath != None:
        with open(savepath, "w") as f:
            f.write(text)
            print(" Clean data saved into ----->%s " % (savepath))

def process_juman_output(juman_outfile):

    print(" Processing juman output ...")
    corpus = []
    hiragana_corpus = []
    daihyou_vocab = {}

    with open(juman_outfile) as csvfile:
        reader = csv.reader(csvfile, delimiter=str(u" "))
        sent = []
        hirag_sent = []
        for line in reader:
            if line[0] == u"@":
                continue

            if line[0] == u"EOS":
                corpus.append(u" ".join(sent))
                hiragana_corpus.append(u" ".join(hirag_sent))
                hirag_sent = []
                sent=[]
                continue

            if line[11] != "NIL":
                value = line[11]
                value = re.sub("代表表記:", u"", value,re.U)
                value = value.split(u"/")[0]
            else:
                value = line[0]

            hiragana = line[1]
            hirag_sent.append(hiragana)

            key = line[0]
            daihyou_vocab[key] = value
            sent.append(key)

        corpus = u"\n".join(corpus)
        hiragana_corpus = u"\n".join(hiragana_corpus)

        corpus = re.sub(u"\n\n",u"\n",corpus)
        hiragana_corpus = re.sub(u"\n\n",u"\n", hiragana_corpus)

    print " All in all unique lemmas: %d" %(len(daihyou_vocab.values()))

    # save a txt corpus file
    with open("data/string_corpus.txt","w") as f:
        for line in corpus.split(u"\n"):
            print >> f, line

    # save hiragana corpus
    with open("data/hiragana_corpus.txt","w") as fo:
        for line in hiragana_corpus.split(u"\n"):
            print >> fo, line

    # save a vocabulary
    with open("data/daihyou_vocab.p", "w") as vocabfile:
        pickle.dump(daihyou_vocab, vocabfile)

    print "cleaning datadir ..."
    subprocess.call('rm -f ./data/juman_input.txt ./data/kytea_out.txt ./data/cleaned_lyrics.txt',
                    shell=True)


def main():
    parser = argparse.ArgumentParser(description="An LSTM language model")
    parser.add_argument('-juman', help='Preprocess juman file', nargs=1)
    parser.add_argument('-crawl', help='Preprocess crawled data', nargs=1)

    opts = parser.parse_args()

    if opts.crawl:
        clean_corpus(opts.crawl[0], savepath="data/juman_input.txt")
        print " Done cleaning crawled data"

    if opts.juman:
        process_juman_output(opts.juman[0])
        print "Done preparing the corpus"

if __name__ == '__main__':
    main()
