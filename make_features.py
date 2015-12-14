# -*- coding: utf-8 -*-
from __future__ import division
import sys
import codecs


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


# 参考までにmain関数をつけておく
def main():
    dummy_fill = u""
    k_prev = 5

    data_path = "data/lyrics_shonan_s27_raw.tsv"
    with open(data_path, "r") as data:
        data_size = sum([1 for _ in data])
    with codecs.open(data_path, "r", encoding="utf-8") as data:
        for i, song in enumerate(data):
            prev_lines = [dummy_fill for _ in xrange(k_prev)]
            # 1行目は飛ばす
            if i == 0:
                continue
            # 要素数が足りない時はcontinue
            temp_split = song.split("\t")
            if len(temp_split) < 2:
                print "too few elements"
                print temp_split
                continue
            # textにtabが含まれていることがあるので" "でjoin
            artist, title, text = temp_split[0], temp_split[1], temp_split[2:]
            text = u" ".join(text)
            lines = text.split(u"<BR>")
            for line in lines:
                line = line.strip()
                if line == "":
                    continue
                line_length = calc_linelength_score(line, prev_lines[-1])
                BoW = calc_BoW_k_score(line, prev_lines, k=1)
                BoW5 = calc_BoW_k_score(line, prev_lines, k=k_prev)

                datum = {"artist": artist,
                         "title": title,
                         "BoW": BoW,
                         "BoW5": BoW5,
                         "line_length": line_length,
                         "orig_line": line
                         }

                prev_lines.append(line)
                if len(prev_lines) > k_prev:
                    del prev_lines[0]

                # print datum["line_length"]

            sys.stderr.write(u"\r %d/%d done" % (i, data_size))
            sys.stderr.flush()

if __name__ == '__main__':
    main()
