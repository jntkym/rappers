#!/share/usr-x86_64/bin/python
# -*- coding: utf-8 -*-

u"""Return a generated lyric

Params:
- seed=<phrase>: seed phrase [optional]
"""

import cgi
import cgitb
import codecs
import sys
from os import path
from os import environ, getpid, remove
import subprocess

DIR_API = path.join(path.dirname(path.abspath(__file__)))
DIR_ROOT = path.join(path.dirname(DIR_API))
DIR_DATA = path.join(DIR_ROOT, 'data')
PID = getpid()
sys.path = [DIR_ROOT,] + sys.path[:]

from NextLine import NextLine

cgitb.enable()
sys.stdout = codecs.getwriter('utf_8')(sys.stdout)

print("Content-type: text/plain; charset=UTF-8\r\n")

if environ['REQUEST_METHOD'] == 'GET':
    form = cgi.FieldStorage()

    seed_phrase = None
    try:
        seed_phrase = form['seed']
    except KeyError:
        pass
    if seed_phrase:
        f_seed = path.join(path.join(DIR_API, 'tmp'), 'lyric_seed.' + str(PID))
        with open(f_seed, 'w') as f:
            f.write(form.getvalue('seed'))
    else:
        f_seed = path.join(DIR_DATA, 'sample_lyric_seed.txt')

    f_model = '/zinnia/huang/rapper/model_2573'
    f_candidates = path.join(DIR_DATA,
                             'sample_nextline_prediction_candidates.txt')
    n_candidates = 100
    for i in range(5):
        svm = NextLine(f_candidates, f_seed,
                       f_model, n_candidates, tmp_dir=DIR_ROOT)
        print(svm.predict())

    if seed_phrase:
        remove(f_seed)
