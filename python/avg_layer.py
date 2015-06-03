import theano
import theano.tensor as T
from keras.layers.core import *

class avg_layer(Layer):
    def __init__(self):
        super(avg_layer,self).__init__()
	self.input = T.matrix()

    def get_output(self, train):
        X = self.get_input(train)
	return T.mean(X, axis = 1)

    def get_config(self):
        return {"name":self.__class__.__name__}
