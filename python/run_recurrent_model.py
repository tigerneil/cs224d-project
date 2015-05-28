from recurrent_model import recurrent_model
from load_word_vectors import load_word_vectors
from utils import *


word_vec_file = "../data/glove/glove.6B.100d.txt"
word_vec_model = "glove"

relations_file = "../data/relations.txt"
train_file = "../data/train/100k/train_100k_dep_path.tsv"
test_file = "../data/test/10k/test_10k_dep_path.csv"


num_words = 400000
word_dim = 100
hidden_dimension = 500
output_dimension = 42
activ = 'sigmoid'
nepochs = 50
batch_size = 100
printevery = 10000
saveevery = 100000
lrate = 0.01
ldecay = 0.000001
reg = 0.01
output_file = "../data/test/test_out/predictions_ep_" + str(nepochs) + "_lr_" + str(lrate) + "_reg_" + str(reg) + "_bs_" + str(batch_size) + "_lrdecay_" + str(ldecay) + ".out"
model_loc = "saved_models/hd_" + str(hidden_dimension) + "_lr_" + str(lrate) + "_reg_" + str(reg) + "_bs_" + str(batch_size) + "_lrdecay_" + str(ldecay)

wv = load_word_vectors(word_vec_file, word_dim)
rel = read_relations(relations_file)
print('Done reading relations and word vectors.')

mod = recurrent_model(wv, rel, activ, num_words, word_dim, hidden_dimension, output_dimension, reg, lrate, ldecay, batch_size, nepochs)
mod.train(train_file, model_loc, printevery, saveevery)
mod.test(test_file, output_file)
