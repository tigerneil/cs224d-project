#-- Parse a line from a python-processed file.
def parseProcessedLine(line):
    relations = []
    # split the line by tab
    vals = line.rstrip('\n').split("\t")
    tokens = vals[0].split(",")
    relations.extend(vals[1].split(","))
    for relation in relations:
	return tokens, relation


# Parse several lines line from a python-processed file.
# lines is a table of lines
def parse_processed_lines(lines):
    token_table = []
    relations_table = []

    for line in lines:
	tokens, relations = parseProcessedLine(line)
	token_table.append(tokens)
        relations_table.append(relations)
    return token_table, relations_table


# Parse a line form a python-processed file.
def parse_test_processed_line(line):
    # split the line by tab
    line = line.rstrip('\n')
    tokens = line.split(",")
    return tokens

#A function to read the relations file and return a table from indices to relation strings
def read_relations(path):
   rel_file = open(path)
   relations_table = []
   
   for line in rel_file:
   	relations_table.append(line.rstrip('\n'))
   return relations_table
