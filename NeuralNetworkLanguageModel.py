# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import yaml
import sys
import codecs

from WordEmbedding import WordEmbedding 

sys.stdin  = codecs.getreader('UTF-8')(sys.stdin)
sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)

class NeuralNetworkLanguageModel:
    def __init__(self, history = 3, lineDim = 13, learningRate = 0.01, iteration = 2):
        self.history = history
        self.lineDim = lineDim
        self.learningRate = learningRate
        self.lineDelimiter = u"||"
        self.wordDelimiter = u":"
        self.iteration = iteration
        self.supervisor_labels_placeholder = tf.placeholder("int32", [None])
        self.input_placeholder = tf.placeholder("float", [None, (self.history + 1)*(self.lineDim)*WordEmbedding.EMBEDDING_SIZE])

    def _getWordVector(self, wordId, word):
        """
        To be written
        """
        with tf.name_scope("word%d" % wordId):
            # Hidden 1
            with tf.name_scope('hidden1'):
                weights = tf.Variable(tf.zeros([WordEmbedding.EMBEDDING_SIZE, 500]), name="weights")
                biases = tf.Variable(tf.zeros([500]), name='biases')
                hidden1 = tf.nn.relu(tf.matmul(word, weights) + biases)

            # Hidden 2
            with tf.name_scope('hidden2'):
                weights = tf.Variable(tf.zeros([500, 500]), name="weights")
                biases = tf.Variable(tf.zeros([500]), name='biases')
                hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases) 

            # Word vector
            with tf.name_scope('wordVector'):
                weights = tf.Variable(tf.zeros([500, WordEmbedding.EMBEDDING_SIZE]), name="weights")
                biases = tf.Variable(tf.zeros([WordEmbedding.EMBEDDING_SIZE]), name='biases')
                wordVector = tf.nn.relu(tf.matmul(hidden2, weights) + biases) 

        return wordVector

    def _getLineVector(self, lineId, line):
        """
        To be written
        """
        with tf.name_scope("line%d" % lineId):
            wordVectors = []
            splitted = tf.split(1, lm.lineDim, line)
            for wordId, word in enumerate(splitted):
                wordVectors.append(self._getWordVector(wordId, word))

            wordVectors = tf.concat(1, wordVectors)

            # Hidden 1
            with tf.name_scope('hidden1'):
                weights = tf.Variable(tf.zeros([wordVectors.get_shape()[1], 500]), name="weights")
                biases = tf.Variable(tf.zeros([500]), name='biases')
                hidden1 = tf.nn.relu(tf.matmul(wordVectors, weights) + biases)

            # Line vector
            with tf.name_scope('lineVector'):
                weights = tf.Variable(tf.zeros([hidden1.get_shape()[1], 500]), name="weights")
                biases = tf.Variable(tf.zeros([500]), name='biases')
                lineVector = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

        return lineVector

    def _getTextVector(self, lines):
        """
        To be written
        """
        lineVectors = []
        for lineIndex, line in enumerate(tf.split(1, lm.history + 1, lines)):
            lineVectors.append(self._getLineVector(lineIndex, line))

        lineVectors = tf.concat(1, lineVectors)
        with tf.name_scope("text"):
            # Hidden 1
            with tf.name_scope('hidden1'):
                weights = tf.Variable(tf.zeros([lineVectors.get_shape()[1], 500]), name="weights") # TODO: what value should I use for stddev
                biases = tf.Variable(tf.zeros([500]), name='biases')
                hidden1 = tf.nn.relu(tf.matmul(lineVectors, weights) + biases)
            
            # Text vector
            with tf.name_scope('textVector'):
                weights = tf.Variable(tf.zeros([hidden1.get_shape()[1], 500]), name="weights") # TODO: what value should I use for stddev
                biases = tf.Variable(tf.zeros([500]), name='biases')
                textVector = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

        return textVector

    def inference(self, lines):
        textVector = self._getTextVector(lines)
        # Linear
        with tf.name_scope('linear'):
            weights = tf.Variable(tf.zeros([textVector.get_shape()[1], 2]), name="weights") # TODO: what value should I use for stddev
            biases = tf.Variable(tf.zeros([2]), name='biases')
            logits = tf.matmul(textVector, weights) + biases
        
        return logits

    def loss(self, logits, labels):
        """Calculates the loss from the logits and the labels.
        Args:
          logits: Logits tensor, float - [batch_size, NUM_CLASSES].
          labels: Labels tensor, int32 - [batch_size].
        Returns:
          loss: Loss tensor of type float.
        """
        # Convert from sparse integer labels in the range [0, NUM_CLASSSES)
        # to 1-hot dense float vectors (that is we will have batch_size vectors,
        # each with NUM_CLASSES values, all of which are 0.0 except there will
        # be a 1.0 in the entry corresponding to the label).
        NUM_CLASSES = 2
        batch_size = tf.size(labels)
        labels = tf.expand_dims(labels, 1)
        indices = tf.expand_dims(tf.range(0, batch_size), 1)
        concated = tf.concat(1, [indices, labels])
        onehot_labels = tf.sparse_to_dense(concated, tf.pack([batch_size, NUM_CLASSES]), 1.0, 0.0)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, onehot_labels, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        return loss

    def training(self, loss):
        trainer = tf.train.GradientDescentOptimizer(self.learningRate).minimize(loss) 
        return trainer

    def getLongVectors(self, trainigData):
        vectors = []
        for datum in trainingData:
            vector = []
            lineList = datum.split(lm.lineDelimiter)
            for line in lineList:
                wordList = line.split(lm.wordDelimiter)
                wordList = ([WordEmbedding.PADDING] * (self.lineDim - len(wordList))) + wordList # Pad if the length of wordList is not enough
                for word in wordList:
                    embed = WordEmbedding(word).vector
                    assert len(embed) == WordEmbedding.EMBEDDING_SIZE, "%d != %d" % (len(embed), WordEmbedding.EMBEDDING_SIZE)
                    vector.append(embed)
            vector = np.asarray(vector).flatten()
            vectors.append(vector)
        vectors = np.asarray(vectors) 
        # print vectors.shape
        return vectors

    def train(self, trainingData, labels, savePath = None):
        trainingData = self.getLongVectors(trainingData)
        feed_dict={self.input_placeholder: trainingData, self.supervisor_labels_placeholder: labels}

        with tf.Session() as sess:
            output = self.inference(self.input_placeholder)
            loss = self.loss(output, self.supervisor_labels_placeholder)
            trainer = self.training(loss)

            init = tf.initialize_all_variables()
            sess.run(init)
            saver = tf.train.Saver()
            for step in xrange(self.iteration):
                sess.run(trainer, feed_dict=feed_dict)
                if step % 100 == 0:
                    print "Loss in iteration %d = %f" % (step + 1, sess.run(loss, feed_dict=feed_dict))
            if savePath:
                save_path = saver.save(sess, "model")
                print "Model saved in file: ", save_path

    def predict(self, data, modelPath):
        data = self.getLongVectors(data)
        saver = tf.train.Saver()
        feed_dict={self.input_placeholder: data}

        with tf.Session() as sess:
            output = self.inference(self.input_placeholder)
            # Restore variables from disk.
            saver.restore(sess, modelPath)
            print "Model restored."
            print sess.run(output, feed_dict=feed_dict)
         
if __name__ == "__main__":
    trainingData = [u"あこんにちは:わたし:の:なまえ:は:にぼじろう:です||こんにちは:わたし:の:なまえ:は:にぼじろう:です||こんにちは:わたし:の:なまえ:は:にぼじろう:です||あなた:が:かの:ゆうめいな:にぼじろう:さん:ですか",
                    u"ちは:わ:の:まえ:は:にろう:です||にちは:たし:の:まえ:は:にう:です||ちは:わたし:の:なまえ:は:うひゃひゃ:うひひ||ほげげ:が:かの:めいな:にぼじろう:さん:でか",
                    u"ちは:わ:の:まえ:は:にろう:です||にちは:たし:の:まえ:は:にう:です||ちは:わたし:の:なまえ:は:うひゃひゃ:うひひ||ほげげ:が:かの:めいな:にぼじろう:さん:で",
                    u"ちは:わ:の:まえ:は:にろう:です||にちは:たし:の:まえ:は:にう:です||ちは:わたし:の:なまえ:は:うひゃひゃ:うひひ||ほげげ:が:かの:めいな:にぼじろう:さん:で"]
    testData = [u"あこんにちは:わたし:の:なまえ:は:にぼじろう:です||こんにちは:わたし:の:なまえ:は:にぼじろう:です||こんにちは:わたし:の:なまえ:は:にぼじろう:です||あなた:が:かの:ゆうめいな:にぼじろう:さん:ですか",
                    u"ちは:わ:の:まえ:は:にろう:です||にちは:たし:の:まえ:は:にう:です||ちは:わたし:の:なまえ:は:うひゃひゃ:うひひ||ほげげ:が:かの:めいな:にぼじろう:さん:でか",
                    u"ちは:わ:の:まえ:は:にろう:です||にちは:たし:の:まえ:は:にう:です||ちは:わたし:の:なまえ:は:うひゃひゃ:うひひ||ほげげ:が:かの:めいな:にぼじろう:さん:で",
                    u"ちは:わ:の:まえ:は:にろう:です||にちは:たし:の:まえ:は:にう:です||ちは:わたし:の:なまえ:は:うひゃひゃ:うひひ||ほげげ:が:かの:めいな:にぼじろう:さん:で"]

    modelPath = "model"
    labels = np.array([1,1,1,1])
    lm = NeuralNetworkLanguageModel()
    lm.train(trainingData, labels, savePath = modelPath)
    lm.predict(testData, modelPath)
