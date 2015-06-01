from recurrent_model import recurrent_model
from load_word_vectors import load_word_vectors
from utils import *
import cPickle

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
nepochs = 10
batch_size = 1000
printevery = 10000
saveevery = 2000000
lrate = 0.001
ldecay = 0.000001
reg = 0.001
output_file = "../data/test/test_out/predictions_ep_" + str(nepochs) + "_lr_" + str(lrate) + "_reg_" + str(reg) + "_bs_" + str(batch_size) + "_lrdecay_" + str(ldecay) + ".out"
model_loc = "saved_models/hd_" + str(hidden_dimension) + "_lr_" + str(lrate) + "_reg_" + str(reg) + "_bs_" + str(batch_size) + "_lrdecay_" + str(ldecay)

wv = load_word_vectors(word_vec_file, word_dim)
#rel = read_relations(relations_file)
rel = None
print('Done reading relations and word vectors.')

init_count = 0
max_f = None
temp = model_loc.split('/')
files = os.listdir("saved_models")
for f in files:
	if temp[len(temp)-1] in f:
		print 'found old file', f
		temp2 = f.split('.')
		temp2 = temp2[len(temp2)-2].split('_')
		temp2 = int(temp2[len(temp2)-1])
		print 'num iterations saved', temp2
		if temp2 > init_count:
			init_count = temp2
			max_f = f
if max_f is not None:
	mod = cPickle.load(open('saved_models/' + max_f, 'rb'))
else:
	mod = recurrent_model(wv, rel, activ, num_words, word_dim, hidden_dimension, output_dimension, reg, lrate, ldecay, batch_size, nepochs)
mod.train(train_file, model_loc, printevery, saveevery, init_count)
mod.test(test_file, output_file)
