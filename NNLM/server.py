import socket
import time
import tensorflow as tf

from NeuralNetworkLanguageModel import *

class NeuralNetworkLanguageModelServer:

    def __init__(self):
        self.lm = NeuralNetworkLanguageModel()
        self.output = None
        self.requestMax = 5
        self.port = 12345
        self.host = '0.0.0.0'

    def predict(self, data, sess):
        if self.output == None:
            raise ValueError("Model is not loaded yet!")

        vectors = self.lm.getLongVectors(data)
        output = sess.run(self.output, feed_dict={self.lm.input_placeholder: vectors}) 
        result = sess.run(tf.argmax(output,1))
        return result

    def _runServer(self, sess):
        # create a socket object
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        
        # get local machine name
        # host = socket.gethostname()
        
        # bind to the port
        serversocket.bind((self.host, self.port))                                  
        
        # queue up to 5 requests
        serversocket.listen(self.requestMax)

        while True:
            # establish a connection
            clientsocket, addr = serversocket.accept()
            print "Got a connection from %s" % str(addr)
            data = "" 
            while True:
                chunk = clientsocket.recv(1024)
                print chunk
                if chunk == "\n": # End of the data
                # if chunk.endswith("\n"):
                    break
                
                data += chunk
            
            lines = data.split("\n")
            result = map(str, self.predict(lines, sess).tolist())
            clientsocket.send(" ".join(result))
            clientsocket.close()

    def runServer(self):
        # Assume that you have 12GB of GPU memory and want to allocate ~4GB:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            with tf.device(self.lm.device):
                self.output = self.lm.inference(self.lm.input_placeholder)
                saver = tf.train.Saver()
                # Restore variables from disk.
                saver.restore(sess, "model")
                print "Model restored."
                self._runServer(sess)

if __name__ == "__main__":
    server = NeuralNetworkLanguageModelServer()
    server.runServer()
