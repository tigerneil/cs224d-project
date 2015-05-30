#!/bin/bash

# set these based on the machine!
export CORE_NLP_DIR=/Users/msushkov/Documents/stanford-corenlp-full-2015-04-20
export GSON_JAR=/Users/msushkov/Documents/gson-2.3.1.jar

CP=$CORE_NLP_DIR/classes:$CORE_NLP_DIR/*:$GSON_JAR:.

javac -classpath $CP ProcessData.java

CURR_FILE=../../../data/train/train_5.tsv

cat $CURR_FILE | python ../process_kbp_data.py | java -classpath $CP ProcessData