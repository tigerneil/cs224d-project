import sys
import csv
import StringIO

'''
Process input from stdin as tsv lines.
Input: string from stdin
Output: string (the same lines as input) to stdout
'''

# Process a single line of TSV and print result to stdout
def process_line(line):
    strio = StringIO.StringIO(line)
    reader = csv.reader(strio, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        gloss = row[0]
        dependencies_conll = row[1]
        words = row[2]
        lemmas = row[3]
        pos_tags = row[4]
        ner_tags = row[5]
        subject_id = row[6]
        subject_entity = row[7]
        subject_link_score = row[8]
        subject_ner = row[9]
        object_id = row[10]
        object_entity = row[11]
        object_link_score = row[12]
        object_ner = row[13]
        subject_begin = int(row[14])
        subject_end = int(row[15])
        object_begin = int(row[16])
        object_end = int(row[17])

        known_relations = None
        incompatible_relations = None
        annotated_relation = None

        # training
        if len(row) > 18:
            known_relations = row[18]
            incompatible_relations = row[19]
            annotated_relation = row[20]

        words = words[1:-1] # remove the first '{' and the last '}'
        words = words.replace('\",\"', '~^~COMMA~^~')
        words_list = words.split(',')
        words_between = None
        if subject_begin < object_begin:
            words_between = words_list[subject_begin : object_end]
        else:
            words_between = words_list[object_begin : subject_end]
        
        relations = ''
        if len(row) > 18:
            known_relations = known_relations[1:-1]
            known_relations_list = known_relations.split(',')
            relations = ','.join(known_relations_list)

        # create a string to return
        output = ','.join(words_between) + "\t" + relations
        print gloss
	print output



#
# MAIN SCRIPT
#

##Script to get input for word2vec
def process_line_word2vec(line):
    strio = StringIO.StringIO(line)
    reader = csv.reader(strio, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        gloss = row[0]
        dependencies_conll = row[1]
        words = row[2]
        lemmas = row[3]
        pos_tags = row[4]
        ner_tags = row[5]
        subject_id = row[6]
        subject_entity = row[7]
        subject_link_score = row[8]
        subject_ner = row[9]
        object_id = row[10]
        object_entity = row[11]
        object_link_score = row[12]
        object_ner = row[13]
        subject_begin = int(row[14])
        subject_end = int(row[15])
        object_begin = int(row[16])
        object_end = int(row[17])

        known_relations = None
        incompatible_relations = None
        annotated_relation = None

        # training
        if len(row) > 18:
            known_relations = row[18]
            incompatible_relations = row[19]
            annotated_relation = row[20]

        #words = words[1:-1] # remove the first '{' and the last '}'
	words = words[1:-1]
	words = words.replace('\",\"', '~^~COMMA~^~')
	words = words.split(",")
        temp = []
	for i, word in enumerate(words):
		cur = word
		if i == subject_begin:
			cur = subject_entity
		if i == object_begin:
			cur = object_entity

		if i > subject_begin and i < subject_end:
			continue
		if i > object_begin and i < object_end:
			continue
		if cur == '~^~COMMA~^~':
			cur = ','
		temp.append(cur)
        
       	output = ' '.join(temp)
        print gloss
	print '--------------------------------------------------------------'
	print output



for line in sys.stdin:
    line = line.strip()
    if line and line != '':
        process_line_word2vec(line)
