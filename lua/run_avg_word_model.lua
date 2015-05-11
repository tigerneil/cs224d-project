require "torch"
require "nn"
util = require "util"
dofile "avg_word_model.lua"
dofile "readWordVectors.lua"

word_vec_file = "../data/glove/glove.6B.100d.txt"
word_vec_model = "glove"

relations_file = "../data/relations.txt"
train_file = "../data/train/1m/train1m_processed.tsv"
test_file = "../data/dev/validation_set_processed.tab"


num_words = 400000
word_dim = 100
hidden_dimension = 500
output_dimension = 42

nepochs = 50
batch_size = 25
printevery = 100000
saveevery = 2000000
lrate = 0.01
ldecay = 1000000
reg = 0.001
output_file = "../data/dev/predictions_ep_"..nepochs .."_lr_" ..lrate .."_reg_" ..reg .."_bs_" ..batch_size ..".out"
use_cuda = false

wv = load_wordVector(word_vec_file, word_dim, word_vec_model)
rel = util.read_relations(relations_file)
print('Done reading relations and word vectors.')

mod = avg_word_model(wv, rel, num_words, word_dim, hidden_dimension, output_dimension, use_cuda)
mod:autotrain(train_file, lrate, ldecay, reg, nepochs, batch_size, printevery, saveevery)
mod:autotest(test_file, output_file)
