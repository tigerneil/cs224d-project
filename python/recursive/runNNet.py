import optparse
import cPickle as pickle

import sgd as optimizer
#from rntn import RNTN
#from rnn2deep import RNN2
from rnn import RNN
import tree as tr
import time
#import matplotlib.pyplot as plt
import numpy as np
import pdb
#from matplotlib.pyplot import *
import os
import joblib

TRAIN_DATA_FILE = os.environ['TRAIN_DATA_FILE']
DEV_DATA_FILE = os.environ['DEV_DATA_FILE']


def run(args=None):
    usage = "usage : %prog [options]"
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--test",action="store_true",dest="test",default=False)

    # Optimizer
    parser.add_option("--minibatch",dest="minibatch",type="int",default=30)
    parser.add_option("--optimizer",dest="optimizer",type="string",
        default="adagrad")
    parser.add_option("--epochs",dest="epochs",type="int",default=50)
    parser.add_option("--step",dest="step",type="float",default=1e-2)


    parser.add_option("--middleDim",dest="middleDim",type="int",default=10)
    parser.add_option("--outputDim",dest="outputDim",type="int",default=5)
    parser.add_option("--wvecDim",dest="wvecDim",type="int",default=30)

    
    parser.add_option("--outFile",dest="outFile",type="string",
        default="models/test.bin")
    parser.add_option("--inFile",dest="inFile",type="string",
        default="models/test.bin")
    parser.add_option("--data",dest="data",type="string",default="train")

    parser.add_option("--model",dest="model",type="string",default="RNN")

    (opts, args) = parser.parse_args(args)


    # make this false if you dont care about your accuracies per epoch, makes things faster!
    evaluate_accuracy_while_training = True

    # Testing
    if opts.test:
        test(opts.inFile, opts.data, opts.model)
        return
    
    print "Loading data..."
    train_accuracies = []
    dev_accuracies = []

    # load training data
    trees = tr.load_trees(TRAIN_DATA_FILE)
    opts.numWords = len(tr.load_word_to_index_map())

    if (opts.model=='RNTN'):
        nn = RNTN(opts.wvecDim,opts.outputDim,opts.numWords,opts.minibatch)
    elif(opts.model=='RNN'):
        nn = RNN(opts.wvecDim,opts.outputDim,opts.numWords,opts.minibatch)
    elif(opts.model=='RNN2'):
        nn = RNN2(opts.wvecDim,opts.middleDim,opts.outputDim,opts.numWords,opts.minibatch)
    else:
        raise '%s is not a valid neural network so far only RNTN, RNN, RNN2' % opts.model
    
    nn.initParams()

    sgd = optimizer.SGD(nn, alpha=opts.step, minibatch=opts.minibatch, optimizer=opts.optimizer)

    dev_trees = tr.load_trees(DEV_DATA_FILE)
    for e in range(opts.epochs):
        start = time.time()
        print "Running epoch %d" % e
        sgd.run(trees)
        end = time.time()
        print "Time per epoch : %f" % (end-start)

        # save the net to the output file
        #f = open(opts.outFile, 'wb')
        #pickle.dump(opts, f, -1)
        #pickle.dump(sgd.costt, f, -1)
        #pickle.dump(nn.stack, f, -1)
        #np.save(f, nn.stack)
        #f.close()
        joblib.dump(opts, opts.outFile + "_opts")
        joblib.dump(sgd.costt, opts.outFile + "_cost")
        joblib.dump(nn.stack, opts.outFile + "_stack")

        if evaluate_accuracy_while_training:
            print "testing on training set..."
            train_accuracies.append(test(opts.outFile, "train", opts.model, trees))
            
            print "testing on dev set..."
            dev_accuracies.append(test(opts.outFile, "dev", opts.model, dev_trees))
            
            # clear the fprop flags in trees and dev_trees
            for tree in trees:
                tr.traverse(tree.root, func=tr.clear_fprop)
            for tree in dev_trees:
                tr.traverse(tree.root, func=tr.clear_fprop)
            print "fprop in trees cleared"


    if False: # don't do this for now
    #if evaluate_accuracy_while_training:
        #print train_accuracies
        #print dev_accuracies
        # Plot train/dev_accuracies here
        x = range(opts.epochs)
        figure(figsize=(6,4))
        plot(x, train_accuracies, color='b', marker='o', linestyle='-', label="training")
        plot(x, dev_accuracies, color='g', marker='o', linestyle='-', label="dev")
        title("Accuracy vs num epochs.")
        xlabel("Epochs")
        ylabel("Accuracy")
        #ylim(ymin=0, ymax=max(1.1*max(train_accuracies),3*min(train_accuracies)))
        legend()
        savefig("train_dev_acc.png")
        #pdb.set_trace()

def test(netFile, dataSet, model='RNN', trees=None):
    if trees == None:
        if dataSet == "train":
            trees = tr.load_trees(TRAIN_DATA_FILE)
        elif dataSet == "dev":
            trees = tr.load_trees(DEV_DATA_FILE)
    
    assert netFile is not None, "Must give model to test"
    print "Testing netFile %s" % netFile

    #f = open(netFile, 'rb')
    #opts = pickle.load(f)
    #_ = pickle.load(f)
    opts = joblib.load(netFile + "_opts")
    _ = joblib.load(netFile + "_cost")
    
    if (model=='RNTN'):
        nn = RNTN(opts.wvecDim,opts.outputDim,opts.numWords,opts.minibatch)
    elif(model=='RNN'):
        nn = RNN(opts.wvecDim,opts.outputDim,opts.numWords,opts.minibatch)
    elif(model=='RNN2'):
        nn = RNN2(opts.wvecDim,opts.middleDim,opts.outputDim,opts.numWords,opts.minibatch)
    else:
        raise '%s is not a valid neural network so far only RNTN, RNN, and RNN2' % opts.model
    
    nn.initParams()
    #nn.stack = pickle.load(f)
    #nn.stack = np.load(f)
    nn.stack = joblib.load(netFile + "_stack")
    #f.close()

    print "Testing %s..." % model

    cost, correct, guess, total = nn.costAndGrad(trees, test=True)
    correct_sum = 0
    for i in xrange(0, len(correct)):        
        correct_sum += (guess[i] == correct[i])
    
    # confusion matrix
    conf_arr = np.zeros((opts.outputDim, opts.outputDim))
    for i in xrange(len(correct)):
        curr_correct = correct[i]
        curr_guess = guess[i]
        conf_arr[curr_correct][curr_guess] += 1.0

    #makeconf(conf_arr)
    
    print "Cost %f, Acc %f" % (cost, correct_sum / float(total))
    return correct_sum / float(total)


def makeconf(conf_arr):
    # makes a confusion matrix plot when provided a matrix conf_arr
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                    interpolation='nearest')

    width = len(conf_arr)
    height = len(conf_arr[0])

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    indexs = '0123456789'
    plt.xticks(range(width), indexs[:width])
    plt.yticks(range(height), indexs[:height])
    # you can save the figure here with:
    plt.savefig("conf_matrix.png")

    plt.show()


if __name__=='__main__':
    run()


