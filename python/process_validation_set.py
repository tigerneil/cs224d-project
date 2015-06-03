import sys

relationsFile = open('../data/dev/relations.txt', 'w')
testFile = open('../data/dev/validation_input.txt', 'w')

def process_line(line):
    relation, subject, obj, words = line.split('\t')
    relationsFile.write(relation+"\n")
    subjIndex = words.find(subject)
    objIndex = words.find(obj)
    if subjIndex == -1 or objIndex == -1:
        print 'Something wrong here'
        return
    if subjIndex < objIndex:
        testLine = words[subjIndex: objIndex+len(obj)]
    else:
        testLine = words[objIndex: subjIndex+len(subject)]
    resultWords = testLine.replace(',', '~^~COMMA~^~').split()
    output = ','.join(resultWords)
    print output
    
    

for line in sys.stdin:
    line = line.strip()
    if line and line != '':
        process_line(line)

relationsFile.close()
