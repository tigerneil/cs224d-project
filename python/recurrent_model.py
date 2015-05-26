import utils
import pickle
import theano
import numpy as np
import theano.tensor as T

#Bi-directional recurrent model, we accumulate the state from both directions along the dependency path and use the concatenated vector for prediction
class recurrent_model:
	def __init__(self, initial_embeddings, relations, activation, num_words, word_dim, state_dimension, output_dimension, regularization, l_r, l_r_decay, batch_size, nepochs):
		self.sdim = state_dimension
		self.odim = output_dimension
		self.wvdim = word_dim
		self.Vdim = num_words
		self.activ = activation
		self.reg = T.scalar(regularization)
		self.lr = T.scalar(l_r)
		self.lrdecay = T.scalar(l_r_decay)
		self.bs = batch_size
		self.ep = nepochs
		# generating lookup table and dictionary from words to lookup table indices from word -> vector maps
		self.word_to_ind = {}
		embed = np.empty((self.Vdim, self.wvdim))
		count = 1
		for word, tensor in initial_embeddings.iteritems():
			self.word_to_ind[word] = count
			embed[count] = initial_embeddings[word]
			count = count + 1
		self.embeddings = theano.shared(embed.astype(theano.config.floatX))
		
        	self.wf = theano.shared(name='wf',value=0.2 * np.random.uniform(-1.0, 1.0, (self.sdim + self.wvdim, self.sdim)).astype(theano.config.floatX))
		self.wb = theano.shared(name='wb',value=0.2 * np.random.uniform(-1.0, 1.0,(self.sdim + self.wvdim, self.sdim)))
        	self.w = theano.shared(name='w', value=0.2 * np.random.uniform(-1.0, 1.0,(self.odim, 2*self.sdim)).astype(theano.config.floatX))
	        self.bf = theano.shared(name='bf',value=np.zeros(self.sdim,dtype=theano.config.floatX))
		self.bb = theano.shared(name='bb',value=np.zeros(self.sdim,dtype=theano.config.floatX))
	        self.b = theano.shared(name='b',value=np.zeros(self.odim,dtype=theano.config.floatX))
	        self.hf = theano.shared(name='hf',value=np.zeros(self.sdim,dtype=theano.config.floatX))
		self.hb = theano.shared(name='hb',value=np.zeros(self.sdim,dtype=theano.config.floatX))
		
		self.dwf = theano.shared(name='wf',value=np.zeros((self.sdim + self.wvdim, self.sdim)).astype(theano.config.floatX))
		self.dwb = theano.shared(name='wb',value=np.zeros((self.sdim + self.wvdim, self.sdim)).astype(theano.config.floatx))
        	self.dw = theano.shared(name='w', value=np.zeros((self.odim, 2*self.sdim)).astype(theano.config.floatX))
	        self.dbf = theano.shared(name='bf',value=np.zeros(self.sdim,dtype=theano.config.floatX))
		self.dbb = theano.shared(name='bb',value=np.zeros(self.sdim,dtype=theano.config.floatX))
	        self.db = theano.shared(name='b',value=np.zeros(self.odim,dtype=theano.config.floatX))
	        self.dhf = theano.shared(name='hf',value=np.zeros(self.sdim,dtype=theano.config.floatX))
		self.dhb = theano.shared(name='hb',value=np.zeros(self.sdim,dtype=theano.config.floatX))	
		#bundling
		self.params = [self.wf, self.wb, self.w, self.bf,self.bb, self.b, self.hf, self.hb]
		self.acc_grads = [self.dwf, self.dwb, self.dw, self.dbf,self.dbb, self.db, self.dhf, self.dhb]

		# Relations to output index
		self.rel_to_ind = {}
		self.ind_to_rel = {}
		for i,rel in enumerate(relations):
			self.rel_to_ind[rel] = i
			self.ind_to_rel[i] = rel

		# Compiling theano functions
		x = T.ivector('x')
		y = T.iscalar('y')
		alpha = T.scalar('alpha')
		cost = self.forwardProp(x, y)
		grads = T.grad(cost, self.params)
		updates = [(acc_grad_i, acc_grad_i - grad_i) for acc_grad_i, grad_i in zip(self.acc_grads, grads)]
		self.step = theano.function([x, y, alpha], cost, updates = updates)
		param_updates = [(param_i, param_i - alpha*grad_i) for param_i, grad_i in zip(self.params, self.acc_grads)]
		param_updates.extend([(grad_i, 0*grad_i) for grad_i in self.acc_grads])
		self.update_params = theano.function([], [], updates = param_updates)
		self.predict = theano.function(x, self.test(x))
		print 'Model initialized'

	def get_fwd_state(self, h_tminus1, x):
		return self.activ(T.sum(T.dot(self.wf, T.concatenate([h_tminus1, self.embeddings[x_t, :]], axis = 0)), self.bf))

	def get_bwd_state(self, h_tplus1, x):
		return self.activ(T.sum(T.dot(self.wb, T.concatenate([h_tplus1, self.embeddings[x_t, :]], axis = 0)), self.bb))

	#Assumes x is a sequence of indices for words
	def forwardProp(self, x, y):
		hforward, _ = theano.scan(fn = self.get_fwd_state, outputs_info = self.hf, sequences = x, n_steps = x.shape[0])
		hbackward, _ = theano.scan(fn = self.get_bwd_state, outputs_info = self.hb, sequences = x[::-1], n_steps = x.shape[0])
		s = T.nnet.softmax(T.sum(T.dot(self.W, T.concatenate([hforward[-1, :], hbackward[-1, :]])), self.b))
		return -T.log(s[y]) + T.mul((T.sum(T.pow(self.wf, 2)) +T.sum(T.pow(self.wb, 2)) + T.sum(T.pow(self.w, 2))), self.reg)
	
	def test(self, x):
		hforward, _ = theano.scan(fn = get_fwd_state, outputs_info = self.hf, sequences = x, n_steps = x.shape[0])
		hbackward, _ = theano.scan(fn = get_bwd_state, outputs_info = self.hb, sequences = x[::-1], n_steps = x.shape[0])
		s = T.nnet.softmax(T.sum(T.dot(self.W, T.concatenate([hforward[-1, :], hbackward[-1, :]])), self.b))
		return T.argmax(s)

	def get_sequence(self, line):
		words = line.split(" ")
		return [self.word_to_ind[word] for word in words]

	def train(self, data_loc, print_every, save_every, reg, lr, bs, ep):
		pass
