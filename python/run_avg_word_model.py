from avg_word_model import avg_word_model
from load_word_vectors import load_word_vectors
from utils import *

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
batch_size = 1
printevery = 100
saveevery = 2000000
lrate = 0.01
ldecay = 1000000
reg = 0.00
output_file = "../data/dev/predictions_ep_" + str(nepochs) + "_lr_" + str(lrate) + "_reg_" + str(reg) + "_bs_" + str(batch_size) + "_lrdecay_" + str(ldecay) + ".out"
model_loc = 'saved_models/'

wv = load_word_vectors(word_vec_file, word_dim)
rel = read_relations(relations_file)
print('Done reading relations and word vectors.')

mod = avg_word_model(wv, rel, num_words, word_dim, hidden_dimension, output_dimension, reg, lrate, ldecay, batch_size, nepochs)
mod.autotrain(train_file, model_loc)
mod.autotest(test_file, output_file)
