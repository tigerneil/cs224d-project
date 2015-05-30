#!/bin/bash

# set these based on the machine!
export CORE_NLP_DIR=/Users/msushkov/Documents/stanford-corenlp-full-2015-04-20
export GSON_JAR=/Users/msushkov/Documents/gson-2.3.1.jar

# if on corn
#export CORE_NLP_DIR=~/stanford-corenlp-full-2015-04-20
#export GSON_JAR=~/gson-2.3.1.jar

CP=$CORE_NLP_DIR/classes:$CORE_NLP_DIR/*:$GSON_JAR:.

# if on corn
#JAVA_PREFIX=/afs/.ir/users/m/s/msushkov/jdk_linux_1.8.0_45/bin/

$JAVA_PREFIXjavac -classpath $CP ProcessData.java

CURR_FILE=../../../data/train/100k/train_100k.tsv

# if on corn
#CURR_FILE=/afs/.ir.stanford.edu/users/m/s/msushkov/cs224d-project/data/train/100k/train_100k.tsv

cat $CURR_FILE | python ../process_kbp_data.py | $JAVA_PREFIXjava -classpath $CP ProcessData > out.txt 