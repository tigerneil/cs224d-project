require "torch"
require "nn"
util = require "util"
dofile "avg_word_model.lua"
dofile "readWordVectors.lua"

word_vec_file = "../data/glove/glove.6B.100d.txt"
word_vec_model = "glove"

relations_file = "../data/relations.txt"
train_file = "../data/train/processed/train_processed.tsv"
test_file = "../data/test_processed.tsv"
output_file = "../data/test_out.txt"

num_words = 400000
word_dim = 100
hidden_dimension = 200
output_dimension = 42

wv = load_wordVector(word_vec_file, word_dim, word_vec_model)
rel = util.read_relations(relations_file)
print('Done reading relations and word vectors.')

mod = avg_word_model(wv, rel, num_words, word_dim, hidden_dimension, output_dimension)
mod:autotrain(train_file, lrate, reg, nepochs, printevery)
mod:autotest(test_file, output_file)