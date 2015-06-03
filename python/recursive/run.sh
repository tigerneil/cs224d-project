#!/bin/bash

# data files
export WORDS_FILE="/juicer/scr82/scr/nlp/data/tac-kbp/tackbp2015/master/tmp/wordVecTrain/words.txt"
export WORD_VECTORS="/juicer/scr82/scr/nlp/data/tac-kbp/tackbp2015/master/tmp/wordVecTrain/trunk/kbp_vectors_new.txt"
export TRAIN_DATA_FILE="/juicer/scr82/scr/nlp/data/tac-kbp/tackbp2015/master/tmp/wordVecTrain/micheal_data/output/100k/combined.txt"
export DEV_DATA_FILE=""

# training params
epochs=48
step=1e-2
wvecDim=300
outputDim=42

# for RNN2 only, otherwise doesnt matter
middleDim=5

model="RNN" # either RNN, RNN2, or RNTN


if [ "$model" == "RNN2" ]; then
    outfile="models/${model}_wvecDim_${wvecDim}_middleDim_${middleDim}_step_${step}_2.bin"
else
    outfile="models/${model}_wvecDim_${wvecDim}_step_${step}_2.bin"
fi

mkdir -p models

echo $outfile


python runNNet.py --step $step --epochs $epochs --outFile $outfile \
                --middleDim $middleDim --outputDim $outputDim --wvecDim $wvecDim --model $model 