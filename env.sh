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

export TEST_DATA_FILE=$curr_dir/data/test.csv

# command to split a large file
#split -l lines file prefix

cat $curr_dir/data/train* > $curr_dir/data/train.tsv

cat $curr_dir/data/train_processed* > $curr_dir/data/train_processed.tsv

export TRAIN_DATA_FILE=$curr_dir/data/train.tsv
export TRAIN_DATA_FILE_PROCESSED=$curr_dir/data/train_processed.tsv

echo "Done."
