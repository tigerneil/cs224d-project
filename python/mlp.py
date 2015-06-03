import utils
import cPickle
import theano
import numpy as np
import theano.tensor as T
import sys
import os

class mlp:
    def __init__(self, initial_embeddings, activation, num_words, word_dim, hdim, odim, reg, l_r, l_r_decay, batch_size, nepochs):
        self.sdim = hdim
        self.odim = odim
        self.wvdim = word_dim
        self.Vdim = num_words
        if activation == 'relu':
            self.activ = lambda x: 0.5*(x + np.abs(x))
        elif activation == 'sigmoid':
            self.activ = T.nnet.sigmoid
        self.reg = reg
        self.lr = l_r
        self.lrdecay = l_r_decay
        self.batch_size = batch_size
        self.ep = nepochs
        # generating lookup table and dictionary from words to lookup table indices from word -> vector maps
        embed = np.zeros((self.Vdim, self.wvdim), dtype ='float32')
        #count = 1
        for i, tensor in initial_embeddings.iteritems():
            embed[i] = initial_embeddings[i]
        self.embeddings = embed
        self.wh1 = theano.shared(name='wh1', value=(0.2 * np.random.uniform(-1.0, 1.0, (2 * self.wvdim, self.sdim))).astype(theano.config.floatX))
        self.b1 = theano.shared(name='b1', value=np.zeros(self.sdim, dtype=theano.config.floatX))
        #self.wh2 = theano.shared(name='wh2', value=(0.2 * np.random.uniform(-1.0, 1.0, (self.sdim, self.wvdim))).astype(theano.config.floatX))
        self.ws = theano.shared(name='ws', value=(0.2 * np.random.uniform(-1.0, 1.0, (self.sdim, self.odim))).astype(theano.config.floatX))
        self.bs = theano.shared(name='bs', value=np.zeros((self.odim), dtype=theano.config.floatX))

        # derivatives
        self.dwh1 = theano.shared(name='dwh1', value=np.zeros((2 * self.wvdim, self.sdim)).astype(theano.config.floatX))
        self.db1 = theano.shared(name='db1', value=np.zeros(self.sdim, dtype=theano.config.floatX))
        #self.dwh2 = theano.shared(name='dwh2', value=np.zeros((self.sdim, self.wvdim)).astype(theano.config.floatX))
        self.dws = theano.shared(name='dws', value=np.zeros((self.sdim, self.odim)).astype(theano.config.floatX))
        self.dbs = theano.shared(name='dbs', value=np.zeros((self.odim), dtype=theano.config.floatX))

        #bundling
        self.params = [self.wh1, self.ws, self.b1, self.bs]
        self.acc_grads = [self.dwh1, self.dws, self.db1, self.dbs]
        
        x = T.fmatrix('x')
        y = T.ivector('y')
        alpha = T.scalar('alpha')
        cost = self.forwardProp(x, y)
        grads = T.grad(cost, self.params)
        updates = [(acc_grad_i, acc_grad_i + alpha*grad_i) for acc_grad_i, grad_i in zip(self.acc_grads, grads)]
        self.step = theano.function([x, y, alpha], cost, updates = updates, allow_input_downcast = True)
        param_updates = [(param_i, param_i - grad_i) for param_i, grad_i in zip(self.params, self.acc_grads)]
        param_updates.extend([(grad_i, 0*grad_i) for grad_i in self.acc_grads])
        self.update_params = theano.function([], [], updates = param_updates)
        self.predict = theano.function([x], self.pred(x), allow_input_downcast = True)
        print 'Model initialized'


    def forwardProp(self, x, y):
        h = self.activ(T.dot(x, self.wh1) + self.b1)
        p = T.transpose(T.dot(h, self.ws) + self.bs)
        s = T.exp(p - T.max(p, axis = 0))
        s = s/T.sum(s, axis = 0)
        return -T.sum(T.log(s[y, T.arange(y.shape[0])])) + T.mul(T.sum(T.pow(self.ws, 2)) + T.sum(T.pow(self.wh1, 2)), self.reg)

    def pred(self, x):
        h = self.activ(T.dot(x, self.wh1) + self.b1)
        p = T.transpose(T.dot(h, self.ws) + self.bs)
        s = T.exp(p - T.max(p, axis = 0))
        s = s/T.sum(s, axis = 0)
        return T.argmax(s, axis=0) + 1

    def preprocess_data(self, data_loc, train = True):
        f = open(data_loc)
        data = []
        for line in f:
            words = [int(word)-1 for word in line.split(" ")]
            rel = None
            if train:
                try:
                    rel = words[0]
                    words = [words[1] , words[len(words) - 1]]
                except e:
                    print e
                    print line
            else:
                try:
                    words = [words[0], words[len(words) - 1]]
                except e:
                    print e
                    print line
            data.append((words, rel))
        return data


    def train(self, data_loc, save_loc, print_every, save_every):
        cost = 0.0
        train_data = self.preprocess_data(data_loc)
        N = len(train_data)
        ind = np.random.randint(0, N-self.batch_size)
        init_count = 0
        max_f = None
        temp = save_loc.split('/')
        files = os.listdir("saved_models")
        for f in files:
            if temp[len(temp)-1] in f:
                if "mlp" not in f:
                    continue
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
            for i,param in enumerate(self.params):
                param.set_value(mod[i])

        for i in range(init_count/self.batch_size, self.ep*N/self.batch_size):
            alpha = self.lr/(1 + self.lrdecay*i*self.batch_size)
            sample = train_data[ind:(ind + self.batch_size)]
            indx = []
            y = []
            for s in sample:
                indx.extend(s[0])
                y.append(s[1])
            temp = self.embeddings[indx]
            x = np.reshape(temp, (self.batch_size, -1))
            y = np.asarray(y)

            cost += self.step(x, y, alpha)
            self.update_params()
            ind = np.random.randint(0, N - self.batch_size)
            if ((i+1)*self.batch_size) % print_every == 0:
                print "Average cost after ", (i+1)*self.batch_size, " iterations is ", cost/print_every
                cost = 0.0
            if ((i+1)*self.batch_size) % save_every == 0:
                temp = []
                for param in self.params:
                    temp.append(param.get_value())
                cPickle.dump(temp, open(save_loc + "_iter_" + str(i*self.batch_size) + ".mlp", 'wb'))

    def test(self, data_loc, out_loc):    
        dat = self.preprocess_data(data_loc, False)
        f = open(out_loc)
        for k,v in dat.iteritems():
            f.write(self.predict(np.reshape(self.embeddings[v[0]], (1, -1))))
            f.write('\n')
