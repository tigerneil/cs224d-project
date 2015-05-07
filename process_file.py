
from process_line import *

filename_in = "data/train.tsv"
filename_out = "data/train_processed.tsv"

file_in = open(filename_in, 'r')
file_out = open(filename_out, 'w')

for line in file_in:
	line = line.strip()
	processed = process_line(line)
	if processed:
		file_out.write(processed + "\n")

file_in.close()
file_out.close()