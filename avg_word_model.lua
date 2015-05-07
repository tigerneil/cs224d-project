require "torch"
require "nn"

local avg_word_model, parent = torch.class('avg_word_model', 'nn.Module')

function avg_word_model:__init(initial_embeddings, hidden_dimension, output_dimension)
   parent.__init(self)
   self.h = hidden_dimension
   self.o = output_dimension
   
   self.model = nn.Sequential()
   -- use the word_embeddings as input, should be in nn.lookuptable format 
   self.model:add(self.getlookuptable(initial_embeddings))

   -- mean of selected words
   self.model:add(nn.Mean(1))

   -- hidden layer
   self.model:add(nn.Linear(self.table.size[2], hidden_dimension))
   self.model:add(nn.Sigmoid())

   -- output layer
   self.model:add(Linear(hidden_dimension, output_dimension))
   self.model:add(nn.SoftMax())
end

function avg_word_model:updateOutput(input)
   self.output = self.model.updateOutput(input)
   return self.output
end

function avg_word_model:updateGradInput(input, gradOutput)
   return self.model:updateGradInput(input, gradOutput)
end

function avg_word_model:accGradParameters(input, gradOutput, scale)
  self.model:accGradParameters(input, gradOutput, scale)
end

-- Add a training function that supports reading training data from disk with negative log likelihood criterion
function avg_word_model:train(lr, lrdecay)

end
-- Add a function to generate lookup table from dictionary
function avg_word_model:get_lookup_table(initial_embeddings)

end



