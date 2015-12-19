#!/share/usr-x86_64/bin/python
# -*- coding: utf-8 -*-

import sys
import subprocess
import cgitb
from os import path

DIR_ROOT = path.join(path.dirname(path.dirname(path.abspath(__file__))))
DIR_DATA = path.join(DIR_ROOT, 'data')

sys.path = [DIR_ROOT,] + sys.path[:]

from NextLine import NextLine

cgitb.enable()


print("Content-type: text/plain; charset=UTF-8\r\n")

f_seed = path.join(DIR_DATA, 'sample_lyric_seed.txt')
f_model = '/zinnia/huang/rapper/model_all'
f_candidates = path.join(DIR_DATA,
                         'sample_nextline_prediction_candidates.txt')
n_candidates = 100
for i in range(5):
    svm = NextLine(f_candidates, f_seed,
                   f_model, n_candidates, tmp_dir=DIR_ROOT)
    print(svm.predict())
