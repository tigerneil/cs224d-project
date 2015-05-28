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
		if activation == 'relu':
			self.activ = lambda x: 0.5*(x + np.abs(x))
		elif activation == 'sigmoid':
			self.activ = T.nnet.sigmoid
		self.reg = regularization
		self.lr = l_r
		self.lrdecay = l_r_decay
		self.bs = batch_size
		self.ep = nepochs
		# generating lookup table and dictionary from words to lookup table indices from word -> vector maps
		self.word_to_ind = {}
		embed = np.zeros((self.Vdim + 1, self.wvdim))
		count = 1
		for word, tensor in initial_embeddings.iteritems():
			self.word_to_ind[word] = count
			embed[count] = initial_embeddings[word]
			count = count + 1
		self.embeddings = theano.shared(name = 'embeddings', value = embed.astype(theano.config.floatX))
		
        	self.wfs = theano.shared(name='wfs',value=(np.eye(self.sdim) + 0.2 * np.random.uniform(-1.0, 1.0, (self.sdim, self.sdim))).astype(theano.config.floatX))
	 	self.wfw = theano.shared(name='wfw',value=0.2 * np.random.uniform(-1.0, 1.0, (self.sdim, self.wvdim)).astype(theano.config.floatX))
		self.wbs = theano.shared(name='wbs',value=(np.eye(self.sdim) + 0.2 * np.random.uniform(-1.0, 1.0,(self.sdim, self.sdim))).astype(theano.config.floatX))
        	self.wbw = theano.shared(name='wbw',value=0.2 * np.random.uniform(-1.0, 1.0, (self.sdim, self.wvdim)).astype(theano.config.floatX))
		self.wf = theano.shared(name='wf', value=0.2 * np.random.uniform(-1.0, 1.0,(self.odim, self.sdim)).astype(theano.config.floatX))
		self.wb = theano.shared(name='wb', value=0.2 * np.random.uniform(-1.0, 1.0,(self.odim, self.sdim)).astype(theano.config.floatX))
		self.bf = theano.shared(name='bf',value=np.zeros(self.sdim,dtype=theano.config.floatX))
		self.bb = theano.shared(name='bb',value=np.zeros(self.sdim,dtype=theano.config.floatX))
        	self.b = theano.shared(name='b',value=np.zeros(self.odim,dtype=theano.config.floatX))
        	self.hf = theano.shared(name='hf',value=np.zeros(self.sdim,dtype=theano.config.floatX))
		self.hb = theano.shared(name='hb',value=np.zeros(self.sdim,dtype=theano.config.floatX))
		
		self.dwfs = theano.shared(name='dwfs',value=np.zeros((self.sdim, self.sdim)).astype(theano.config.floatX))
		self.dwbs = theano.shared(name='dwbs',value=np.zeros((self.sdim, self.sdim)).astype(theano.config.floatX))
		self.dwfw = theano.shared(name='dwfw',value=np.zeros((self.sdim, self.wvdim)).astype(theano.config.floatX))
		self.dwbw = theano.shared(name='dwbw',value=np.zeros((self.sdim, self.wvdim)).astype(theano.config.floatX))
		self.dwf = theano.shared(name='dwf', value=np.zeros((self.odim, self.sdim)).astype(theano.config.floatX))
		self.dwb = theano.shared(name='dwb', value=np.zeros((self.odim, self.sdim)).astype(theano.config.floatX))
		self.dbf = theano.shared(name='dbf',value=np.zeros(self.sdim,dtype=theano.config.floatX))
		self.dbb = theano.shared(name='dbb',value=np.zeros(self.sdim,dtype=theano.config.floatX))
        	self.db = theano.shared(name='db',value=np.zeros(self.odim,dtype=theano.config.floatX))
        	self.dhf = theano.shared(name='dhf',value=np.zeros(self.sdim,dtype=theano.config.floatX))
		self.dhb = theano.shared(name='dhb',value=np.zeros(self.sdim,dtype=theano.config.floatX))	
		#bundling
		self.params = [self.wfs, self.wfw, self.wbs, self.wbw, self.wf, self.wb, self.bf,self.bb, self.b, self.hf, self.hb]
		self.acc_grads = [self.dwfs, self.dwfw, self.dwbs, self.dwbw, self.dwf, self.dwb, self.dbf,self.dbb, self.db, self.dhf, self.dhb]

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
		updates = [(acc_grad_i, acc_grad_i + alpha*grad_i) for acc_grad_i, grad_i in zip(self.acc_grads, grads)]
		self.step = theano.function([x, y, alpha], cost, updates = updates, allow_input_downcast = True)
		param_updates = [(param_i, param_i - grad_i) for param_i, grad_i in zip(self.params, self.acc_grads)]
		param_updates.extend([(grad_i, 0*grad_i) for grad_i in self.acc_grads])
		self.update_params = theano.function([], [], updates = param_updates)
		self.predict = theano.function([x], self.pred(x), allow_input_downcast = True)
		print 'Model initialized'

	def get_fwd_state(self, x, h_tminus1):
		p1 = T.dot(self.wfs, h_tminus1) 
		p2 = T.dot(self.wfw, self.embeddings[x])
		p3 = p1 + p2 + self.bf
		return self.activ(p3)

	def get_bwd_state(self, x, h_tplus1):
		p1 = T.dot(self.wbs, h_tplus1) 
		p2 = T.dot(self.wbw, self.embeddings[x])
		p3 = p1 + p2 + self.bb
		return self.activ(p3)

	#Assumes x is a sequence of indices for words
	def forwardProp(self, x, y):
		hforward, _ = theano.scan(fn = self.get_fwd_state, outputs_info = self.hf, sequences = x, n_steps = x.shape[0])
		hbackward, _ = theano.scan(fn = self.get_bwd_state, outputs_info = self.hb, sequences = x[::-1], n_steps = x.shape[0])
		p1 = T.dot(self.wf, hforward[-1])
		p2 = T.dot(self.wb, hbackward[-1])
		p3 = p1 + p2 + self.b
		s = T.exp(p3 - T.max(p3))
		s = s/T.sum(s)
		return -T.log(s[y]) + T.mul(T.sum(T.pow(self.wfs, 2)) +T.sum(T.pow(self.wbs, 2)) + T.sum(T.pow(self.wbw, 2)) + T.sum(T.pow(self.wfw, 2)) + T.sum(T.pow(self.wf, 2)) + T.sum(T.pow(self.wb, 2)), self.reg)
	
	def pred(self, x):
		hforward, _ = theano.scan(fn = self.get_fwd_state, outputs_info = self.hf, sequences = x, n_steps = x.shape[0])
		hbackward, _ = theano.scan(fn = self.get_bwd_state, outputs_info = self.hb, sequences = x[::-1], n_steps = x.shape[0])
		p1 = T.dot(self.wf, hforward[-1])
		p2 = T.dot(self.wb, hbackward[-1])
		p3 = p1 + p2 + self.b
		s = T.exp(p3 - T.max(p3))
		s = s/T.sum(s) 
		return T.argmax(s, axis = 0)

	def preprocess_data(self, data_loc, train = True):
		f = open(data_loc)
		data = []
		for line in f:
			row = line.split("\t")
			words = np.asarray([self.word_to_ind.get(word, self.Vdim) for word in row[0].split(" ")])
			rel = None
			if train:
				try:
					rel = self.rel_to_ind[row[1].strip()]
				except:
					print row
			
			data.append((words, rel))
		return data

	def train(self, data_loc, save_loc, print_every, save_every):
		cost = 0.0
		train_data = self.preprocess_data(data_loc)
		ind = 0
		N = len(train_data)
		for i in range(self.ep*N):
			count = i+1
			alpha = self.lr/(1 + self.lrdecay*count)
			cost += self.step(train_data[ind][0], train_data[ind][1], alpha)
			ind = (ind + 1) % N
			if count % self.bs ==0:
				self.update_params()
				ind = np.random.randint(0, N)
			if count % print_every == 0:
				print "Average cost after ", count, " iterations is ", cost/count
			if count % save_every == 0:
				pickle.dump(self, open(save_loc + "_iter_" + str(count) + ".rnn", 'w'))


	def test(self, data_loc, out_loc):
		dat = self.preprocess_data(data_loc, False)
		f = open(out_loc)
		for k,v in dat.iteritems():
			f.write(self.ind_to_rel(self.predict(v[0])))
			f.write('\n')
