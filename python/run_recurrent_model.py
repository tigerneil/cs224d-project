from recurrent_model import recurrent_model
from load_word_vectors import load_word_vectors
from utils import *


word_vec_file = "../data/word2vec_300_new.txt"
word_vec_model = "new"

relations_file = ""
train_file = "../data/kbp_train.txt"
test_file = "../data/kbp_dev.txt"


num_words = 2675308
word_dim = 300
hidden_dimension = 300
output_dimension = 42
activ = 'sigmoid'
nepochs = 50
batch_size = 100
printevery = 1000
saveevery = 100000
lrate = 0.001
ldecay = 0.000001
reg = 0.01
output_file = "../data/test/test_out/predictions_ep_" + str(nepochs) + "_lr_" + str(lrate) + "_reg_" + str(reg) + "_bs_" + str(batch_size) + "_lrdecay_" + str(ldecay) + ".out"
model_loc = "saved_models/hd_" + str(hidden_dimension) + "_lr_" + str(lrate) + "_reg_" + str(reg) + "_bs_" + str(batch_size) + "_lrdecay_" + str(ldecay)

wv = load_word_vectors(word_vec_file, word_dim)
#rel = read_relations(relations_file)
rel = None
print('Done reading relations and word vectors.')

mod = recurrent_model(wv, rel, activ, num_words, word_dim, hidden_dimension, output_dimension, reg, lrate, ldecay, batch_size, nepochs)
mod.train(train_file, model_loc, printevery, saveevery)
mod.test(test_file, output_file)
