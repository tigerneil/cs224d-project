import utils
import pickle
import theano
import numpy as np
import theano.tensor as T

#Bi-directional recurrent model, we accumulate the state from both directions along the dependency path and use the concatenated vector for prediction
class recurrent_model:
	def __init__(self, initial_embeddings, relations, num_words, word_dim, state_dimension, output_dimension):
		self.sdim = state_dimension
		self.odim = output_dimension
		self.wvdim = word_dim
		self.Vdim = num_words
		
		# generating lookup table and dictionary from words to lookup table indices from word -> vector maps
		self.word_to_ind = {}
		embed = np.empty((self.Vdim, self.wvdim))
		count = 1
		for word, tensor in initial_embeddings.iteritems():
			self.word_to_ind[word] = count
			embed[count] = initial_embeddings[word]
			count = count + 1
		self.embeddings = theano.shared(embed.astype(theano.config.floatX))
		
        	self.wf = theano.shared(name='wf',value=0.2 * numpy.random.uniform(-1.0, 1.0, (self.sdim + self.wvdim, self.sdim)).astype(theano.config.floatX))
		self.wb = theano.shared(name='wb',value=0.2 * numpy.random.uniform(-1.0, 1.0,(self.sdim + self.wvdim, self.sdim)))
        	self.w = theano.shared(name='w', value=0.2 * numpy.random.uniform(-1.0, 1.0,(self.odim, 2*self.sdim)).astype(theano.config.floatX))
	        self.bf = theano.shared(name='bf',value=numpy.zeros(self.sdim,dtype=theano.config.floatX))
		self.bb = theano.shared(name='bb',value=numpy.zeros(self.sdim,dtype=theano.config.floatX))
	        self.b = theano.shared(name='b',value=numpy.zeros(self.odim,dtype=theano.config.floatX))
	        self.hf = theano.shared(name='hf',value=numpy.zeros(self.sdim,dtype=theano.config.floatX))
		self.hb = theano.shared(name='hb',value=numpy.zeros(self.sdim,dtype=theano.config.floatX))
		
		#bundling
		self.params = [self.wf, self.wb, self.w, self.bf,
                       self.bb, self.b, self.hf, self.hb]
		
		# Relations to output index
		self.rel_to_ind = {}
		self.ind_to_rel = {}
		for i,rel in enumerate(relations):
			self.rel_to_ind[rel] = i
			self.ind_to_rel[i] = rel
		print 'Model initialized'

	def get_fwd_state(h_tminus1, x):
		return T.nnet.sigmoid(T.sum(T.dot(self.wf, T.concatenate([h_tminus1, self.embeddings[x_t, :]], axis = 0)), self.bf))

	def get_bwd_state(h_tplus1, x):
		return T.nnet.sigmoid(T.sum(T.dot(self.wb, T.concatenate([h_tplus1, self.embeddings[x_t, :]], axis = 0)), self.bb))

	#Assumes x is a sequence of indices for words
	def forwardProp(x, y):
		hforward, _ = theano.scan(fn = get_fwd_state, outputs_info = self.hf, sequences = x, n_steps = x.shape[0])
		hbackward, _ = theano.scan(fn = get_bwd_state, outputs_info = self.hb, sequences = x[::-1], n_steps = x.shape[0])
		s = T.nnet.softmax(T.sum(T.dot(self.W, T.concatenate([hforward[-1, :], hbackward[-1, :]])), self.b))
		return -T.log(s[y])
	
	def predict(x):
		hforward, _ = theano.scan(fn = get_fwd_state, outputs_info = self.hf, sequences = x, n_steps = x.shape[0])
		hbackward, _ = theano.scan(fn = get_bwd_state, outputs_info = self.hb, sequences = x[::-1], n_steps = x.shape[0])
		s = T.nnet.softmax(T.sum(T.dot(self.W, T.concatenate([hforward[-1, :], hbackward[-1, :]])), self.b))
		return T.argmax(s)

	def get_sequence(line):
		words = line.split(" ")
		return [self.word_to_ind[word] for word in words]


