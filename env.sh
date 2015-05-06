#! /bin/bash

#
# NOTE: this script must be executed from the directory where it lives
#

curr_dir="$( pwd )"

###
### GLOVE
###

# glove on Wikipedia 2014 + gigaword
glove_50_file="http://www-nlp.stanford.edu/data/glove.6B.50d.txt.gz"
glove_100_file="http://www-nlp.stanford.edu/data/glove.6B.100d.txt.gz"
glove_200_file="http://www-nlp.stanford.edu/data/glove.6B.200d.txt.gz"
glove_300_file="http://www-nlp.stanford.edu/data/glove.6B.300d.txt.gz"

export GLOVE_50=$curr_dir/data/glove/glove.6B.50d.txt
export GLOVE_100=$curr_dir/data/glove/glove.6B.100d.txt
export GLOVE_200=$curr_dir/data/glove/glove.6B.200d.txt
export GLOVE_300=$curr_dir/data/glove/glove.6B.300d.txt

# download glove files if they are missing
if [ ! -f "$GLOVE_50" ]; then 
	cd data/glove
	wget $glove_50_file
	gunzip glove.6B.50d.txt.gz
	cd ../..
fi

if [ ! -f "$GLOVE_100" ]; then 
	cd data/glove
	wget $glove_100_file
	gunzip glove.6B.100d.txt.gz
	cd ../..
fi

if [ ! -f "$GLOVE_200" ]; then 
	cd data/glove
	wget $glove_200_file
	gunzip glove.6B.200d.txt.gz
	cd ../..
fi

if [ ! -f "$GLOVE_300" ]; then 
	cd data/glove
	wget $glove_300_file
	gunzip glove.6B.300d.txt.gz
	cd ../..
fi


###
### SAMPLE KBP DATA DUMP
###

export KBP_10k=$curr_dir/data/data10k.csv

echo "Done."