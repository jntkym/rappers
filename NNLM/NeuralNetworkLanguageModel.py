# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import codecs
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
    def __init__(self, configPath="config.yml"):
        with codecs.open(configPath, "r", "UTF-8") as f:
            configString = f.read()
            data = yaml.load(configString) 

        self.history = data["history"]
        self.lineDim = data["lineDim"]
        self.batchSize = data["batchSize"]
        self.learningRate = data["learningRate"]
        self.lineDelimiter = data["lineDelimiter"]
        self.wordDelimiter = data["wordDelimiter"]
        self.iteration = data["iteration"]
        self.device = data["device"]
        self.supervisor_labels_placeholder = tf.placeholder("int32", [None])
        self.input_placeholder = tf.placeholder(
                "float", 
                [None, (self.history + 1)*(self.lineDim)*WordEmbedding.EMBEDDING_SIZE]
        )
        self.layerSize = data["layerSize"]

    def _getWordVector(self, wordId, word):
        """
        To be written
        """
        with tf.name_scope("word%d" % wordId):
            # Hidden 1
            with tf.name_scope('hidden1'):
                weights = tf.Variable(
                        tf.truncated_normal([WordEmbedding.EMBEDDING_SIZE, self.layerSize["word"]["hidden1"]], stddev = 0.1), 
                        name="weights"
                        )
                biases = tf.Variable(
                        tf.random_normal([self.layerSize["word"]["hidden1"]]), 
                        name='biases'
                        )
                hidden1 = tf.nn.relu(tf.add(tf.matmul(word, weights), biases))

            # Hidden 2
            with tf.name_scope('hidden2'):
                weights = tf.Variable(
                        tf.truncated_normal([self.layerSize["word"]["hidden1"], self.layerSize["word"]["hidden2"]], stddev = 0.1), 
                        name="weights"
                        )
                biases = tf.Variable(
                        tf.random_normal([self.layerSize["word"]["hidden2"]]), 
                        name='biases'
                        )
                hidden2 = tf.nn.relu(tf.add(tf.matmul(hidden1, weights), biases))

            # Word vector
            with tf.name_scope('wordVector'):
                weights = tf.Variable(
                        tf.truncated_normal([self.layerSize["word"]["hidden2"], WordEmbedding.EMBEDDING_SIZE], stddev = 0.1), 
                        name="weights"
                        )
                biases = tf.Variable(
                        tf.random_normal([WordEmbedding.EMBEDDING_SIZE]), 
                        name='biases'
                        )
                wordVector = tf.nn.relu(tf.add(tf.matmul(hidden2, weights), biases))

        return wordVector

    def _getLineVector(self, lineId, line):
        """
        To be written
        """
        with tf.name_scope("line%d" % lineId):
            wordVectors = []
            splitted = tf.split(1, self.lineDim, line)
            for wordId, word in enumerate(splitted):
                wordVectors.append(self._getWordVector(wordId, word))

            wordVectors = tf.concat(1, wordVectors)
            lineVectorSize = (self.lineDim)*WordEmbedding.EMBEDDING_SIZE
            # Hidden 1
            with tf.name_scope('hidden1'):
                weights = tf.Variable(
                        tf.truncated_normal([lineVectorSize, self.layerSize["line"]["hidden1"]], stddev = 0.1), 
                        name="weights"
                        )
                biases = tf.Variable(
                        tf.random_normal([self.layerSize["line"]["hidden1"]]), 
                        name='biases'
                        )
                hidden1 = tf.nn.relu(tf.add(tf.matmul(wordVectors, weights), biases))

            # Line vector
            with tf.name_scope('lineVector'):
                weights = tf.Variable(
                        tf.truncated_normal([self.layerSize["line"]["hidden1"], lineVectorSize], stddev = 0.1), 
                        name="weights"
                        )
                biases = tf.Variable(
                        tf.random_normal([lineVectorSize]), 
                        name='biases'
                        )
                lineVector = tf.nn.relu(tf.add(tf.matmul(hidden1, weights), biases))

        return lineVector

    def _getTextVector(self, lines):
        """
        To be written
        """
        lineVectors = []
        for lineIndex, line in enumerate(tf.split(1, self.history + 1, lines)):
            lineVectors.append(self._getLineVector(lineIndex, line))

        lineVectors = tf.concat(1, lineVectors)
        textVectorSize = (self.history + 1) * (self.lineDim)*WordEmbedding.EMBEDDING_SIZE

        with tf.name_scope("text"):
            # Hidden 1
            with tf.name_scope('hidden1'):
                weights = tf.Variable(
                        tf.truncated_normal([textVectorSize, self.layerSize["text"]["hidden1"]], stddev = 0.1), 
                        name="weights"
                        )
                biases = tf.Variable(
                        tf.random_normal([self.layerSize["text"]["hidden1"]]), 
                        name='biases'
                        )
                hidden1 = tf.nn.relu(tf.add(tf.matmul(lineVectors, weights), biases))
            
            # Text vector
            with tf.name_scope('textVector'):
                weights = tf.Variable(
                        tf.truncated_normal([self.layerSize["text"]["hidden1"], textVectorSize], stddev = 0.1), 
                        name="weights"
                        )
                biases = tf.Variable(
                        tf.random_normal([textVectorSize]), 
                        name='biases'
                        )
                textVector = tf.nn.relu(tf.add(tf.matmul(hidden1, weights), biases))

        return textVector

    def inference(self, lines):
        textVector = self._getTextVector(lines)
        textVectorSize = (self.history + 1) * (self.lineDim)*WordEmbedding.EMBEDDING_SIZE
        # Linear
        with tf.name_scope('linear'):
            weights = tf.Variable(
                    tf.truncated_normal([textVectorSize, 2], stddev = 0.1), 
                    name="weights"
                    )
            biases = tf.Variable(
                    tf.random_normal([2]), 
                    name='biases'
                    )
            logits = tf.add(tf.matmul(textVector, weights), biases)
        
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
        onehot_labels = tf.sparse_to_dense(
                concated, 
                tf.pack([batch_size, NUM_CLASSES]), 
                1.0, 
                0.0
                )
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, onehot_labels, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        return loss

    def training(self, loss):
        trainer = tf.train.AdagradOptimizer(self.learningRate).minimize(loss) 
        return trainer

    def getLongVectors(self, trainingData):
        vectors = []
        for datum in trainingData:
            vector = []
            lineList = datum.split(self.lineDelimiter)
            for line in lineList:
                wordList = line.split(self.wordDelimiter)
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
        # Assume that you have 12GB of GPU memory and want to allocate ~4GB:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            with tf.device(self.device):
                output = self.inference(self.input_placeholder)
                loss = self.loss(output, self.supervisor_labels_placeholder)
                trainer = self.training(loss)
                numExample = len(trainingData)
                totalBatch = int(numExample/self.batchSize)
                init = tf.initialize_all_variables()
                sess.run(init)
                saver = tf.train.Saver()
                for step in xrange(self.iteration):
                    averageCost = 0.0
                    for _input, _labels in self.getPermutatedBatch(trainingData, labels):
                        feed_dict={self.input_placeholder: _input, self.supervisor_labels_placeholder: _labels}
                        sess.run(trainer, feed_dict=feed_dict)
                        averageCost += sess.run(loss, feed_dict=feed_dict)
                    if step % 1 == 0:
                        print "Loss in iteration %d = %f" % (step + 1, averageCost/totalBatch)
                if savePath:
                    save_path = saver.save(sess, savePath)
                    print "Model saved in file: ", save_path

    def getPermutatedBatch(self, data, labels):
        data = np.array(data)
        numExample = data.shape[0]
        assert numExample == labels.shape[0]
        totalBatch = int(numExample/self.batchSize)
        perm = np.arange(numExample)
        np.random.shuffle(perm)
        data = data[perm]
        labels = labels[perm]
        curIndex = 0
        for i in xrange(totalBatch):
            batchData = self.getLongVectors(data[curIndex:curIndex + self.batchSize])
            yield batchData, labels[curIndex:curIndex + self.batchSize]
            curIndex += self.batchSize

    def predict(self, data, modelPath):
        data = self.getLongVectors(data)
        feed_dict={self.input_placeholder: data}

        # Assume that you have 12GB of GPU memory and want to allocate ~4GB:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            with tf.device(self.device):
                output = self.inference(self.input_placeholder)
                saver = tf.train.Saver()
                # Restore variables from disk.
                saver.restore(sess, modelPath)
                print "Model restored."
                output = sess.run(output, feed_dict=feed_dict)
                result = sess.run(tf.argmax(output,1))
                return result
