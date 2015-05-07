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
	
	-- use the lookup table generated sbove, should be in nn.lookuptable format 
	self.model:add(self.lookup)

	-- mean of selected words
	self.model:add(nn.Mean(1))

	-- hidden layer
	self.model:add(nn.Linear(self.wvdim, self.hdim))
	self.model:add(nn.Sigmoid())

	-- output layer
	self.model:add(nn.Linear(self.hdim, self.odim))
	self.model:add(nn.SoftMax())
end

function avg_word_model:updateOutput(input)
	self.output = self.model:updateOutput(input)
	return self.output
end

function avg_word_model:updateGradInput(words, gradOutput)
	return self.model:updateGradInput(input, gradOutput)
end

function avg_word_model:accGradParameters(words, gradOutput, scale)
	self.model:accGradParameters(input, gradOutput, scale)
end

-- Add a training function that supports reading training data from disk with negative log likelihood criterion
function avg_word_model:autotrain(data_loc, lr, lrdecay)

end

