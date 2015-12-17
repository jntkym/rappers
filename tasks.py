# -*- coding: utf-8 -*-
from invoke import run, task 
# import codecs
import sys

from NeuralNetworkLanguageModel import * 
import DataSet

sys.stdin  = codecs.getreader('UTF-8')(sys.stdin)
sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)

@task
def clean():
    run("rm *.pyc")

@task
def train(modelPath, trainingDataPath):
    trainingData = []
    labels = []
    with codecs.open(trainingDataPath, "r", "UTF-8") as f:
        for line in f:
            line = line.strip()
            input, label = line.split("\t") 
            trainingData.append(input)
            labels.append(int(label))
    # labels = [0,1,0,0]
    # trainingData = [u"あこんにちは:わたし:の:なまえ:は:にぼじろう:です||こんにちは:わたし:の:なまえ:は:にぼじろう:です||こんにちは:わたし:の:なまえ:は:にぼじろう:です||あなた:が:かの:ゆうめいな:にぼじろう:さん:ですか",
    #                 u"ちは:わ:の:まえ:は:にろう:です||にちは:たし:の:まえ:は:にう:です||ちは:わたし:の:なまえ:は:うひゃひゃ:うひひ||ほげげ:が:かの:めいな:にぼじろう:さん:でか",
    #                 u"ちは:わ:の:まえ:は:にろう:です||にちは:たし:の:まえ:は:にう:です||ちは:わたし:の:なまえ:は:うひゃひゃ:うひひ||ほげげ:が:かの:めいな:にぼじろう:さん:で",
    #                 u"ちは:わ:の:まえ:は:にろう:です||にちは:たし:の:まえ:は:にう:です||ちは:わたし:の:なまえ:は:うひゃひゃ:うひひ||ほげげ:が:かの:めいな:にぼじろう:さん:で"]
    labels = np.array(labels)
    lm = NeuralNetworkLanguageModel()
    lm.train(trainingData, labels, savePath = modelPath)

@task
def test(modelPath, testDataPath):
    testData = []
    labels = []
    with codecs.open(testDataPath, "r", "UTF-8") as f:
        for line in f:
            line = line.strip()
            input, label = line.split("\t") 
            testData.append(input)

    # testData = [u"あこんにちは:わたし:の:なまえ:は:にぼじろう:です||こんにちは:わたし:の:なまえ:は:にぼじろう:です||こんにちは:わたし:の:なまえ:は:にぼじろう:です||あなた:が:かの:ゆうめいな:にぼじろう:さん:ですか",
    #                 u"ちは:わ:の:まえ:は:にろう:です||にちは:たし:の:まえ:は:にう:です||ちは:わたし:の:なまえ:は:うひゃひゃ:うひひ||ほげげ:が:かの:めいな:にぼじろう:さん:でか",
    #                 u"ちは:わ:の:まえ:は:にろう:です||にちは:たし:の:まえ:は:にう:です||ちは:わたし:の:なまえ:は:うひゃひゃ:うひひ||ほげげ:が:かの:めいな:にぼじろう:さん:で",
    #                 u"ちは:わ:の:まえ:は:にろう:です||にちは:たし:の:まえ:は:にう:です||ちは:わたし:の:なまえ:は:うひゃひゃ:うひひ||ほげげ:が:かの:めいな:にぼじろう:さん:で"]
    lm = NeuralNetworkLanguageModel()
    print lm.predict(testData, modelPath)

@task
def createData(inputFilename, outputFilename):
    DataSet.createData(inputFilename, outputFilename)
