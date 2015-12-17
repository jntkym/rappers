# -*- coding: utf-8 -*-
from invoke import run, task 
# import codecs
import sys

from NeuralNetworkLanguageModel import * 
import generateDataForNN5 as gen

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

    lm = NeuralNetworkLanguageModel()
    print lm.predict(testData, modelPath)

@task
def createData(inputFilename, outputFilename):
    gen.createData(inputFilename, outputFilename)
