#
# Processes a data file in csv format
#

import os

DATA_FILE = os.environ['DATA_FILE']


def process_file(filename):
	f = open(filename, 'r')
	for line in f:
		process_line(line)
	f.close


def process_line(line):
	line = line.strip()

	# TODO: extract each column from this line, then use dd and get_dep_path
	