# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import numpy as np
import sys
import yaml
import argparse
from random import randint
import codecs

sys.stdin  = codecs.getreader('UTF-8')(sys.stdin)
sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)

songs = []

def fetchFalseLine(songIndex, lineIndex):
    assert len(songs) > 0
    cnt = 0
    while True:
        falseSongIndex = randint(0, len(songs) - 1)
        falseLineIndex = randint(0, len(songs[falseSongIndex]) - 1)
        if (falseSongIndex != songIndex or falseLineIndex != lineIndex) and len(songs[falseSongIndex][falseLineIndex]) > 1:
            return songs[falseSongIndex][falseLineIndex]
        cnt += 1
        if cnt >= 100:
            sys.exit("Cannot find different line from the current line!!!")

def createData(inputFilename, outputFilename, configPath="config.yml"):
    with codecs.open(configPath, "r", "UTF-8") as f:
        configString = f.read()
        data = yaml.load(configString) 

    history = data["history"]
    lineDim = data["lineDim"]
    padding = data["padding"] 
    lineDelimiter = data["lineDelimiter"]
    wordDelimiter = data["wordDelimiter"]

    with codecs.open(inputFilename, "r", "UTF-8") as f:
        song = [padding]*history
        for line in f:
            line = line.strip()
            if line == u"":
                if len(song) > 0:
                    songs.append(song)
                    song = [padding] * history
                    
                continue
    
            words = line.split(u" ")
            if len(words) < lineDim:
                words = [padding]*(lineDim - len(words)) + words
            elif len(words) > lineDim:
                words = words[len(words) - lineDim:]
    
            song.append(words)
    
        if len(song) > 0:
            songs.append(song)
    
    out = codecs.open(outputFilename, "w", "UTF-8")
    for songIndex, song in enumerate(songs):
        for lineIndex in xrange(history, len(song), 2):
            lines = lineDelimiter.join(map(lambda x: wordDelimiter.join(x), song[lineIndex - history:lineIndex]) + [wordDelimiter.join(song[lineIndex])])
            out.write("%s\t1\n" % lines)

        for lineIndex in xrange(history + 1, len(song), 2):
            falseLine = fetchFalseLine(songIndex, lineIndex)
            lines = lineDelimiter.join(map(lambda x: wordDelimiter.join(x), song[lineIndex - history:lineIndex]) + [wordDelimiter.join(falseLine)])
            out.write("%s\t0\n" % lines)

    out.close()

if __name__ == "__main__":
    createData("data/NN_input.txt","output")
