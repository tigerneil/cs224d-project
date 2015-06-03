import numpy as np
import collections
import pdb
from tree import load_word_vectors

# This is a 2-Layer Deep Recursive Neural Netowrk with two ReLU Layers and a softmax layer

def relu(x):
    return np.maximum(x, 0)

def softmax(x):
    input_x = x
    
    if len(x.shape) == 1:
        x = x.reshape((1, x.shape[0]))
    
    row_maxes = np.amax(x, axis=1).reshape((x.shape[0], 1))
    x = np.exp(x - row_maxes)
    row_sums = np.sum(x, axis=1).reshape((x.shape[0], 1))
    x = x / row_sums
    x = x.reshape(input_x.shape)
    
    return x


class RNN2:

    def __init__(self,wvecDim, middleDim, outputDim,numWords,mbSize=30,rho=1e-4):
        self.wvecDim = wvecDim
        self.outputDim = outputDim
        self.middleDim = middleDim
        self.numWords = numWords
        self.mbSize = mbSize
        self.defaultVec = lambda : np.zeros((wvecDim,))
        self.rho = rho

    def initParams(self):
        np.random.seed(12341)

        # Word vectors
        #self.L = 0.01*np.random.randn(self.wvecDim,self.numWords)
        self.L = load_word_vectors().T # shape is (num_words, word_dim)

        # Hidden activation weights for layer 1
        self.W1 = 0.01*np.random.randn(self.wvecDim,2*self.wvecDim)
        self.b1 = np.zeros((self.wvecDim))

        # Hidden activation weights for layer 2
        self.W2 = 0.01*np.random.randn(self.middleDim,self.wvecDim)
        self.b2 = np.zeros((self.middleDim))

        # Softmax weights
        self.Ws = 0.01*np.random.randn(self.outputDim,self.middleDim) # note this is " U " in the notes and the handout.. there is a reason for the change in notation
        self.bs = np.zeros((self.outputDim))

        self.stack = [self.L, self.W1, self.b1, self.W2, self.b2, self.Ws, self.bs]

        # Gradients
        self.dW1 = np.empty(self.W1.shape)
        self.db1 = np.empty((self.wvecDim))
        
        self.dW2 = np.empty(self.W2.shape)
        self.db2 = np.empty((self.middleDim))

        self.dWs = np.empty(self.Ws.shape)
        self.dbs = np.empty((self.outputDim))

    def costAndGrad(self,mbdata,test=False): 
        """
        Each datum in the minibatch is a tree.
        Forward prop each tree.
        Backprop each tree.
        Returns
           cost
           Gradient w.r.t. W1, W2, Ws, b1, b2, bs
           Gradient w.r.t. L in sparse form.

        or if in test mode
        Returns 
           cost, correctArray, guessArray, total
        """
        cost = 0.0
        correct = []
        guess = []
        total = 0.0

        self.L, self.W1, self.b1, self.W2, self.b2, self.Ws, self.bs = self.stack
        # Zero gradients
        self.dW1[:] = 0
        self.db1[:] = 0
        
        self.dW2[:] = 0
        self.db2[:] = 0

        self.dWs[:] = 0
        self.dbs[:] = 0
        self.dL = collections.defaultdict(self.defaultVec)

        # Forward prop each tree in minibatch
        for tree in mbdata: 
            # do forward propagation (basically the end result of this is root.hActs1)
            self.forwardProp(tree.root)

            # compute the probabilities only at the tree level, not at the node level
            tree.root.hActs2 = relu(np.dot(self.W2, tree.root.hActs1) + self.b2)
            y_hat = softmax(np.dot(self.Ws, tree.root.hActs2) + self.bs)
            tree.probs = y_hat
            cost -= np.log(y_hat[tree.relation_label])
            correct.append(tree.relation_label)
            guess.append(np.argmax(y_hat))  

            total += 1
        if test:
            return (1./len(mbdata))*cost,correct,guess,total

        # Back prop each tree in minibatch
        for tree in mbdata:
            # error from root is y_hat - y
            delta5 = tree.probs
            delta5[tree.relation_label] -= 1.0

            self.dbs += delta5

            # dU
            self.dWs += np.outer(delta5, tree.root.hActs2)

            delta4 = np.dot(self.Ws.T, delta5)
            delta3 = delta4 * (tree.root.hActs2 > 0)

            self.db2 += delta3
            self.dW2 += np.outer(delta3, tree.root.hActs1)

            delta2 = np.dot(self.W2.T, delta3)

            # the error from the relation label doesn't get added anywhere else besides the root
            self.backProp(tree.root, delta2)

        # scale cost and grad by mb size
        scale = (1./self.mbSize)
        for v in self.dL.itervalues():
            v *=scale
        
        # Add L2 Regularization 
        cost += (self.rho/2)*np.sum(self.W1**2)
        cost += (self.rho/2)*np.sum(self.W2**2)
        cost += (self.rho/2)*np.sum(self.Ws**2)

        return scale*cost,[self.dL,scale*(self.dW1 + self.rho*self.W1),scale*self.db1,
                                   scale*(self.dW2 + self.rho*self.W2),scale*self.db2,
                                   scale*(self.dWs+self.rho*self.Ws),scale*self.dbs]

    # same as fprop in rnn.py since the "deep" layer is only added at the label, which happens
    # outside of this method at the top of the root
    def forwardProp(self, node):
        # if this node has only 1 child, just pass through
        if len(node.children) == 1:
            self.forwardProp(node.children[0])
            node.hActs1 = node.children[0].hActs1
        # if we are in a leaf node, set hActs1 to be the word vector
        elif node.isLeaf:
            node.hActs1 = self.L[:, node.word_index] # shape is (10,)
        else:
            assert len(node.children) == 2
            left = node.children[0]
            right = node.children[1]

            # if haven't finished doing forward prop on the first child, do it
            if not left.fprop:
                self.forwardProp(left)
            # if haven't finished doing forward prop on the second child, do it
            if not right.fprop:
                self.forwardProp(right)

            # both children now have hActs1
            node.hActs1 = relu(np.dot(self.W1, np.hstack([left.hActs1, right.hActs1])) + self.b1)
        
        node.fprop = True


    # same as backprop in rnn.py since the "deep" layer is only added at the label, which happens
    # outside of this method at the top of the root
    def backProp(self, node, delta2):
        # Clear nodes
        node.fprop = False

        # if this node has only 1 child, just pass through
        if len(node.children) == 1:
            self.backProp(node.children[0], delta2)
        else:
            if node.isLeaf:
                self.dL[node.word_index] += delta2
            else:
                assert len(node.children) == 2 
                left = node.children[0]
                right = node.children[1]

                delta1 = delta2 * (node.hActs1 > 0)
                self.db1 += delta1
                self.dW1 += np.outer(delta1, np.hstack([left.hActs1, right.hActs1]))
                delta0 = np.dot(self.W1.T, delta1)
                self.backProp(left, delta0[:self.wvecDim])
                self.backProp(right, delta0[self.wvecDim:])
        
        
    def updateParams(self,scale,update,log=False):
        """
        Updates parameters as
        p := p - scale * update.
        If log is true, prints root mean square of parameter
        and update.
        """
        if log:
            for P,dP in zip(self.stack[1:],update[1:]):
                pRMS = np.sqrt(np.mean(P**2))
                dpRMS = np.sqrt(np.mean((scale*dP)**2))
                print "weight rms=%f -- update rms=%f"%(pRMS,dpRMS)

        self.stack[1:] = [P+scale*dP for P,dP in zip(self.stack[1:],update[1:])]

        # handle dictionary update sparsely
        dL = update[0]
        for j in dL.iterkeys():
            self.L[:,j] += scale*dL[j]

    def toFile(self,fid):
        import cPickle as pickle
        pickle.dump(self.stack,fid)

    def fromFile(self,fid):
        import cPickle as pickle
        self.stack = pickle.load(fid)

    def check_grad(self,data,epsilon=1e-6):

        cost, grad = self.costAndGrad(data)

        err1 = 0.0
        count = 0.0
        print "Checking dWs, dW1 and dW2..."
        for W,dW in zip(self.stack[1:],grad[1:]):
            W = W[...,None] # add dimension since bias is flat
            dW = dW[...,None] 
            for i in xrange(W.shape[0]):
                for j in xrange(W.shape[1]):
                    W[i,j] += epsilon
                    costP,_ = self.costAndGrad(data)
                    W[i,j] -= epsilon
                    numGrad = (costP - cost)/epsilon
                    err = np.abs(dW[i,j] - numGrad)
                    err1+=err
                    count+=1
        if 0.001 > err1/count:
            print "Grad Check Passed for dW"
        else:
            print "Grad Check Failed for dW: Sum of Error = %.9f" % (err1/count)
        # check dL separately since dict
        dL = grad[0]
        L = self.stack[0]
        err2 = 0.0
        count = 0.0
        print "Checking dL..."
        for j in dL.iterkeys():
            for i in xrange(L.shape[0]):
                L[i,j] += epsilon
                costP,_ = self.costAndGrad(data)
                L[i,j] -= epsilon
                numGrad = (costP - cost)/epsilon
                err = np.abs(dL[j][i] - numGrad)
                err2+=err
                count+=1

        if 0.001 > err2/count:
            print "Grad Check Passed for dL"
        else:
            print "Grad Check Failed for dL: Sum of Error = %.9f" % (err2/count)

if __name__ == '__main__':
    import tree as treeM
    train = treeM.load_trees()
    numW = len(treeM.load_word_to_index_map())

    wvecDim = 4
    middleDim = 4
    outputDim = 42

    rnn = RNN2(wvecDim, middleDim, outputDim, numW, mbSize=4)
    rnn.initParams()

    mbData = train[:4]
    
    print "Numerical gradient check..."
    rnn.check_grad(mbData)






