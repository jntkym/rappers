# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import yaml
import sys
import codecs

# from WordEmbedding import WordEmbedding 

sys.stdin  = codecs.getreader('UTF-8')(sys.stdin)
sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)

class NeuralNetworkLanguageModel:
    def __init__(self, history = 3, lineDim = 13, learningRate = 0.01):
        self.history = history
        self.lineDim = lineDim
        self.learningRate = learningRate
        self.lineDelimiter = u"||"
        self.wordDelimiter = u":"

    def _getWordVector(self, word, wordId):
        """
        To be written
        """
        with tf.name_scope("word%d" % wordId):
            embedding = WordEmbedding(word)
            # Hidden 1
            with tf.name_scope('hidden1'):
                weights = tf.Variable(tf.random_normal([300, 500], stddev=0.35), name="weights") # TODO: what value should I use for stddev
                biases = tf.Variable(tf.zeros([500]), name='biases')
                hidden1 = tf.nn.relu(tf.matmul(embedding, weights) + biases)

            # Hidden 2
            with tf.name_scope('hidden2'):
                weights = tf.Variable(tf.random_normal([500, 500], stddev=0.35), name="weights") # TODO: what value should I use for stddev
                biases = tf.Variable(tf.zeros([500]), name='biases')
                hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases) 

            # Word vector
            with tf.name_scope('wordVector'):
                weights = tf.Variable(tf.random_normal([500, 300], stddev=0.35), name="weights") # TODO: what value should I use for stddev
                biases = tf.Variable(tf.zeros([300]), name='biases')
                wordVector = tf.nn.relu(tf.matmul(hidden2, weights) + biases) 

        return wordVector

    def _getLineVector(self, lineId, line):
        """
        To be written
        """
        wordVectors = []
        for wordId, word in enumerate(line):
            wordVectors.append(self.getWordVector(wordId, word))

        wordVectors = tf.concat(0, wordVectors)

        with tf.name_scope("line%d" % lineId):
            # Hidden 1
            with tf.name_scope('hidden1'):
                weights = tf.Variable(tf.random_normal([300, 500], stddev=0.35), name="weights") # TODO: what value should I use for stddev
                biases = tf.Variable(tf.zeros([500]), name='biases')
                hidden1 = tf.nn.relu(tf.matmul(wordVectors, weights) + biases)

            # Line vector
            with tf.name_scope('lineVector'):
                weights = tf.Variable(tf.random_normal([300, 500], stddev=0.35), name="weights") # TODO: what value should I use for stddev
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

        lineVectors = tf.concat(0, lineVectors)
        with tf.name_scope("text"):
            # Hidden 1
            with tf.name_scope('hidden1'):
                weights = tf.Variable(tf.random_normal([300, 500], stddev=0.35), name="weights") # TODO: what value should I use for stddev
                biases = tf.Variable(tf.zeros([500]), name='biases')
                hidden1 = tf.nn.relu(tf.matmul(lineVector, weights) + biases)
            
            # Text vector
            with tf.name_scope('textVector'):
                weights = tf.Variable(tf.random_normal([300, 500], stddev=0.35), name="weights") # TODO: what value should I use for stddev
                biases = tf.Variable(tf.zeros([500]), name='biases')
                textVector = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

        return textVector

    def inference(self, lines):
        textVector = self._getTextVector(lines)
        # Linear
        with tf.name_scope('linear'):
            weights = tf.Variable(tf.random_normal([300, 2], stddev=0.35), name="weights") # TODO: what value should I use for stddev
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

    def training(self):
        trainer = tf.train.GradientDescentOptimizer(self.learningRate).minimize(self.loss) 
        return trainer

    def train(self, trainingData, labels):
        trainingData = [map(lambda x: x.split(lm.wordDelimiter), data.split(lm.lineDelimiter)) for data in trainingData]
        # TODO: convertTrainingDataToLongVectors()
        supervisor_labels_placeholder = tf.placeholder("string", [None, 2])
        input_placeholder = tf.placeholder("string", [None, (lm.history + 1)*(lm.lineDim)*WordEmbedding.dim])
        feed_dict={input_placeholder: trainingData, supervisor_labels_placeholder: labels}

        with tf.Session() as sess:
            output = lm.inference(input_placeholder)
            loss = lm.loss(output, supervisor_labels_placeholder)
            trainer = lm.training(loss)

        init = tf.initialize_all_variables()
        sess.run(init)

        for step in range(1000):
            sess.run(trainer, feed_dict=feed_dict)
            # if step % 100 == 0:
                # print sess.run(loss, feed_dict=feed_dict)
         
if __name__ == "__main__":
    trainingData = [u"こんにちは:わたし:の:なまえ:は:にぼじろう:です||こんにちは:わたし:の:なまえ:は:にぼじろう:です||こんにちは:わたし:の:なまえ:は:にぼじろう:です||あなた:が:かの:ゆうめいな:にぼじろう:さん:ですか",
                    u"こんにちは:わたし:の:なまえ:は:にぼじろう:です||こんにちは:わたし:の:なまえ:は:にぼじろう:です||こんにちは:わたし:の:なまえ:は:うひゃひゃ:うひひ||ほげげ:が:かの:ゆうめいな:にぼじろう:さん:ですか"]
    labels = np.array([0,1])
    lm = NeuralNetworkLanguageModel()
    lm.train(trainingData, labels)

