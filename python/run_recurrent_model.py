from recurrent_model import recurrent_model
from load_word_vectors import load_word_vectors
from utils import *
import cPickle
import os
import sys

word_vec_file = "../data/word2vec_300_new.txt"
word_vec_model = "new"

relations_file = ""
train_file = "../data/train_data_dir/recurrent_model_train_data_sorted.ssv"
test_file = "../data/dev_gold/kbp_format_dev.txt"


num_words = 2675308
word_dim = 300
hidden_dimension = 600
output_dimension = 42
activ = 'sigmoid'
nepochs = 20
batch_size = 1000
printevery = 5000
saveevery = 500000
lrate = 0.001
ldecay = 1e-6
reg = 0.2
output_file = "../data/test/test_out/predictions_ep_" + str(nepochs) + "_lr_" + str(lrate) + "_reg_" + str(reg) + "_bs_" + str(batch_size) + "_lrdecay_" + str(ldecay) + ".out"
model_loc = "saved_models/hd_" + str(hidden_dimension) + "_lr_" + str(lrate) + "_reg_" + str(reg) + "_bs_" + str(batch_size) + "_lrdecay_" + str(ldecay)

wv = load_word_vectors(word_vec_file, word_dim)
#rel = read_relations(relations_file)
rel = None
print('Done reading relations and word vectors.')

mod = recurrent_model(wv, rel, activ, num_words, word_dim, hidden_dimension, output_dimension, reg, lrate, ldecay, batch_size, nepochs)
mod.train(train_file, model_loc, printevery, saveevery)
mod.test(test_file, output_file)
