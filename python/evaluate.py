import sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("labels", help="The file containing the gold set to evalute with")

parser.add_argument("predictions", help="The file containing the predictions by the model")

args = parser.parse_args()

labelFile = open(args.labels, 'r')

predictionFile = open(args.predictions, 'r')

relStats = {}
relCounts = {}

numCorrect = 0
total = 0
for label, prediction in zip(labelFile, predictionFile):
    if label == prediction:
        numCorrect += 1
        relStats[label] = relStats.get(label, 0) + 1
    relCounts[label] = relCounts.get(label, 0) + 1
    
    total += 1

print numCorrect, total
print 'Accuracy is: ', numCorrect/ float(total)
print relCounts
for k in relStats.keys():
    print k, relStats[k]/float(relCounts[k])
