require "torch"
require "nn"
local utils = require("util")

local avg_word_model, parent = torch.class('avg_word_model', 'nn.Module')
function avg_word_model:__init(initial_embeddings, num_words, word_dim, hidden_dimension, output_dimension)
	parent.__init(self)
	self.hdim = hidden_dimension
	self.odim = output_dimension
	self.wvdim = word_dim
	self.Vdim = num_words

	-- generating lookup table and dictionary from words to lookup table indices from word -> vector maps
	self.word_to_ind = {}
	self.lookup = nn.LookupTable(self.Vdim + 1, self.wvdim)
	count = 1
	for word, tensor in pairs(initial_embeddings) do
		self.word_to_ind[word] = count
		self.lookup.weight[count] = tensor
		count = count + 1
	end
	self.lookup.weight[count] = torch.Tensor(self.wvdim):fill(0)
	utils.setDefault(self.word_to_ind, count)
	
	
	self.model = nn.Sequential()
	
	-- use the lookup table generated above, should be in nn.lookuptable format 
	self.model:add(self.lookup)

	-- mean of selected words
	self.model:add(nn.Mean(1))

	-- hidden layer
	self.model:add(nn.Linear(self.wvdim, self.hdim))
	self.model:add(nn.Sigmoid())

	-- output layer
	self.model:add(nn.Linear(self.hdim, self.odim))
	self.model:add(nn.LogSoftMax())
	self.reg_layers = {self.model:get(3).weight, self.model:get(5).weight}
	self.criterion = nn.ClassNLLCriterion()  

end

-- Add a training function that supports reading training data from disk with negative log likelihood criterion
function avg_word_model:autotrain(data_loc, lr, reg)
	--Iterate the next lines over the dataset
	local path = data_loc
    	local inputFile = io.open(path)

    	local line = inputFile:read("*l")
    	while line do
       		sgd_step(line, lr, reg)
       		line = inputFile:read("*l")
    	end
end

function avg_word_model:sgd_step(line, lr, reg)
	-- Get nn input and output from line
	
	self.criterion:forward(self.model:forward(input), output)
	-- (1) zero the accumulation of the gradients
	self.model:zeroGradParameters()
	-- (2) accumulate gradients
	self.model:backward(input, self.criterion:backward(self.model.output, output))
	-- (3) update parameters with a 0.01 learning rate
	self.model:updateParameters(lr)

	-- Adding regularization for all linear layers, ignoring the word vectors
	for _,w in ipairs(self.reg_layers) do
		w:add(-lr*reg, w)
	end
end

-- Add a testing function that reads test data from disk and outputs predictions
function avg_word_model:autotest(data_loc)
	
end

-- function to convert table of words to longtensor of indices
function avg_word_model:words_to_indices(words)
	indices = {}
	for i, word in ipairs(words) do
		table.insert(indices, self.word_to_ind[word])
	end
	return torch.LongTensor(indices)
end
