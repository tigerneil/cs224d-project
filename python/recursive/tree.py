import collections
import cPickle as pickle
import os
import re
import pdb
import numpy as np

#WORDS_FILE = '/juicer/scr82/scr/nlp/data/tac-kbp/tackbp2015/master/tmp/wordVecTrain/words.txt'
#WORD_VECTORS = '/juicer/scr82/scr/nlp/data/tac-kbp/tackbp2015/master/tmp/wordVecTrain/trunk/kbp_vectors_new.txt'
#WORD_VECTORS = 'vectors.txt'
#WORDS_FILE = 'words.txt'
#TRAIN_DATA_FILE = 'java/out.txt'

WORD_VECTORS = os.environ['WORD_VECTORS']
WORDS_FILE = os.environ['WORDS_FILE']
TRAIN_DATA_FILE = os.environ['TRAIN_DATA_FILE']

REGEX = '( )|(\()|(\))'
UNK = 'UUNNKK'

class Node: # a node in the tree
    def __init__(self, label=None, word_index=None):
        self.label = label # the label for a given node (NP, JJ, etc. - not the relation)
        self.word_index = word_index # NOT a word vector, but index into L.. i.e. wvec = L[:,node.word]
        self.parent = None # reference to parent
        self.children = []
        self.isLeaf = False # true if I am a leaf (could have probably derived this from if I have a word)
        self.fprop = False # true if we have finished performing fowardprop on this node (note, there are many ways to implement the recursion.. some might not require this flag)
        self.hActs1 = None # h1 from the handout
        self.hActs2 = None # h2 from the handout (only used for RNN2)

# Each Tree has a single relation as a label.
class Tree:
    # tree_string is something like (NP (NP (JJ chief) (@NP (NN scientist) (NNS 10044_Squyres))) (PP (IN of) (NP (NN Cornell_University))))
    # label is an int (index of relation)
    def __init__(self, tree_string, relation_label):
        self.relation_label = relation_label
        self.probs = None # y_hat
        self.tree_string = tree_string

        tokens = [x for x in re.split(REGEX, tree_string) if x is not None and x.strip() != '']
        self.root = self.parse(tokens)

    def tree_to_string(self):
        return tree_to_string(self.root, '')

    # example tree:
    # (NP (NP (JJ chief) (@NP (NN scientist) (NNS 10044_Squyres))) (PP (IN of) (NP (NN Cornell_University))))
    def parse(self, tokens, parent=None):
        if len(tokens) == 0:
            return None

        split = 2 # position after open and label
        count_open = count_close = 0

        if tokens[split] == '(': 
            count_open += 1
            split += 1
        # Find where left child and right child split
        while count_open != count_close:
            if tokens[split] == '(':
                count_open += 1
            if tokens[split] == ')':
                count_close += 1
            split += 1

        node = Node(tokens[1].strip())
        node.parent = parent 

        # leaf Node
        if count_open == 0:
            node.word_index = ''.join(tokens[2:-1])
            node.isLeaf = True
            return node

        left_child = self.parse(tokens[2:split], parent=node)
        right_child = self.parse(tokens[split:-1], parent=node)
        if left_child is not None:
            node.children.append(left_child)
        if right_child is not None:
            node.children.append(right_child)

        return node


# (NP (NP (JJ chief) (@NP (NN scientist) (NNS 10044_Squyres))) (PP (IN of) (NP (NN Cornell_University))))
def tree_to_string(node, string_repr):
    string_repr += '('
    string_repr += node.label
    string_repr += ' '
    if node.isLeaf:
        string_repr += str(node.word)
        string_repr += ')'
    else:
        for i, child in enumerate(node.children):
            string_repr += tree_to_string(child, '')
            
            # if we are not at the last index
            # (insert spaces between the children but not after the last one)
            if i != len(node.children) - 1:
                string_repr += ' '
            else:
                string_repr += ')'
    return string_repr

# DFS on a tree
def traverse(root, func=None, args=None):
    func(root, args)
    for child in root.children:
        traverse(child, func, args)

def clear_fprop(node, words):
    node.fprop = False

def map_words(node, word_to_index_map):
    if node.isLeaf:
        if node.word_index not in word_to_index_map:
            node.word_index = word_to_index_map[UNK]
        else:
            node.word_index = word_to_index_map[node.word_index]

# Returns a 2D numpy array of shape (num_words, word_dim)
def load_word_vectors(filename=WORD_VECTORS):
    embeddings = []
    dim = -1
    with open(filename) as f:
        for i, line in enumerate(f):
            # skip the first line
            if i == 0:
                continue

            temp = line.split(" ")
            arr = np.array(map(float, temp[0:len(temp)]))
            dim = arr.shape[0]
            embeddings.append(arr)

    # create a word vector for UNK
    unk_vec = np.random.uniform(0.0, 0.1, (dim,))
    embeddings.append(unk_vec)

    return np.array(embeddings)

def load_word_to_index_map(filename=WORDS_FILE):
    word_to_index = {}
    c = 0
    with open(filename, 'r') as f:
        for line in f:
            word = line.strip()
            word_to_index[word] = c
            c += 1
    word_to_index[UNK] = c
    return word_to_index

def load_trees(filename=TRAIN_DATA_FILE):
    word_to_index_map = load_word_to_index_map()
    trees = []
    count = 0
    with open(filename, 'r') as f:
        for line in f:
            if count % 1000 == 0:
                print "Processed %d lines..." % count
                count += 1

            vals = line.strip().rsplit('\t', 1) # split on the last tab in this line
            tree_string = vals[0]
            relations = vals[1].split(',')
            for rel in relations:
                curr_tree = Tree(tree_string, int(rel))
                trees.append(curr_tree)

    # instead of the actual words at the leaves, use their indices into the word map
    for tree in trees:
        traverse(tree.root, func=map_words, args=word_to_index_map)

    print "Loaded %d datapoints." % len(trees)
    
    return trees
      
if __name__=='__main__':
    train_trees = load_trees()
