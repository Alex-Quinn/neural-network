#!/bin/sh

cd ~/code/CS1675/neural-network

for i in `seq 1 10`;
do
    echo k=$i >> test_results_no_hidden.txt
    python digit_classifier_no_hidden.py data/train/train_$i data/test/test_$i >> test_results_no_hidden.txt
    echo >> test_results_no_hidden.txt
    echo >> test_results_no_hidden.txt
done

for i in `seq 1 10`;
do
    echo k=$i >> test_results_one_hidden.txt
    python digit_classifier_one_hidden.py data/train/train_$i data/test/test_$i >> test_results_one_hidden.txt
    echo >> test_results_one_hidden.txt
    echo >> test_results_one_hidden.txt
done

for i in `seq 1 10`;
do
    echo k=$i >> test_results_custom.txt
    python digit_classifier_custom.py data/train/train_$i data/test/test_$i >> test_results_custom.txt
    echo >> test_results_custom.txt
    echo >> test_results_custom.txt
done