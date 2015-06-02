import numpy as np

def load_word_vectors(path, dim):
	print "Loading word vectors from ", path
	embeddings = {}
	#count = 0
	#print "Printing first 5 examples"
	
	#with open(path) as f:
	#	for line in f:
	#		temp = line.split(" ")
	#		embeddings[temp[0]] = np.array(map(float, temp[1:len(temp)]))
			#if count < 5:
			#	print temp[0], embeddings[temp[0]]
			#count += 1
	with open(path) as f:
		for i,line in enumerate(f):
			temp = line.split(" ")
			embeddings[i] = np.array(map(float, temp[0:len(temp)]))

	return embeddings
