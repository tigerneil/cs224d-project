from keras.layers.core import *
from keras.layers.embeddings import *
from keras.models import *
from keras.optimizers import SGD
import numpy as np
from avg_layer import avg_layer


class avg_word_model:
	def __init__(self, initial_embeddings, relations, num_words, word_dim, hidden_dimension, output_dimension, reg, alpha, lrdecay, bs, ep)
		self.hdim = hidden_dimension
		self.odim = output_dimension
		self.wvdim = word_dim
		self.Vdim = num_words + 1
		self.reg = reg
		self.lr = alpha
		self.lrdecay = lrdecay
		self.batch_size = bs
		self.nepochs = ep

		# generating lookup table and dictionary from words to lookup table indices from word -> vector maps
		self.word_to_ind = {}
		embed = np.array((self.Vdim, self.wvdim))
		count = 0
		for word, tensor in initial_embeddings.iteritems():
			self.word_to_ind[word] = count
			embed[count, :] = initial_embeddings[word]
			count = count + 1
		embed[count] = np.random(self.wvdim)
		
		self.model = Sequential()
		
		# use the lookup table generated above to initialize embeddings 
		self.model.add(Embedding(self.Vdim, self.wvdim, init = 'uniform', weights = embed))

		# mean of selected words
		self.model.add(avg_layer())

		# hidden layer
		self.model.add(Dense(self.wvdim, self.hdim, W_regularizer = self.reg))
		self.model.add(Activation('sigmoid'))

		# output layer
		self.model.add(Dense(self.hdim, self.odim, W_regularizer = self.reg))
		self.model.add(Activation('softmax'))
	
		sgd = SGD(lr = self.lr, decay = self.lrdecay)
		self.model.compile(loss = 'categorical_crossentropy', optimizer = sgd)
		
		# Relations to output index
		self.rel_to_ind = {}
		self.ind_to_rel = {}
		for i,rel in enumerate(relations):
			self.rel_to_ind[rel] = i
			self.ind_to_rel[i] = rel
		print 'Model initialized'
	end

	#Add a training function that supports reading training data from disk with negative log likelihood criterion
	def autotrain(self, data_loc, printevery, saveevery):
		#Iterate the next lines over the dataset
		count = 0
		for i in range(self.nepochs):
			inputFile = open(data_loc)

			batch_of_lines = []
			cost = 0.0
			for line in inputFile:
				batch_of_lines.append(line)
				count = count + 1

				if count % printevery == 0:
					print("Read ", count," lines")
					print("Total cost over training samples ", cost)
					cost = 0.0

				if count % self.batch_size == 0:
					cost += self.batch_sgd_step(self, batch_of_lines, self.lr/(1 + self.lrdecay*count))
					batch_of_lines = []

				if count % saveevery == 0:
					self.save_model(count)

	def batch_sgd_step(self, lines):
		# Get nn input and output from line batch
		token_list, relation_list = utils.parse_processed_lines(lines)
		inp = []
		for tokens in token_list:
			inp.append([self.word_to_ind[token] for token in tokens])
		inp = np.asarray(inp)
		rel = [self.rel_to_ind[relation] for relation in relation_list]
		out = np.zeros((len(lines), self.odim))
		out[range(len(lines)), rel] = 1
		out = np.asarray(out)
		hist = self.model.fit(inp, out, batch_size = len(lines), nb_epoch = 1)
		return hist['loss']*len(lines)

	# Add a testing function that reads test data from disk and outputs predictions
	def autotest(data_loc, out_path, num_lines):
		#Iterate the next lines over the dataset
		inputFile = open(data_loc)

		outFile = open(out_path, "w")
		out = 0
		for line in inputFile:
			out = self.predict(self, line)
			outFile.write(out)
			outFile.write('\n')

	def predict(line):
		#Get nn input and output from line
		tokens = utils.parse_test_processed_line(line)
		tmp = [self.word_to_ind[token] for token in tokens]
		inp = [tmp]
		return self.ind_to_rel[self.model.predict_classes(inp)[0]]
	end

