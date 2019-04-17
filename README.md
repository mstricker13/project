# project

## Current State:

badStuff: A tutorial I tried out at first, but it wasn't good enough to build upon.

torchImpl: A better LSTM for Seq2Seq machine translation but uses torch

kerasImpl: Should be a keras implementation of torchImpl

kerasImplTimeSerias: Adapted the Keras implementation for time series

## To do:
* Fix Bug with different horizons in csv file?
* Add Networks with GRU's
* Fix a bug in a network
* make for all datasets --> change dataset into CIF format, handle it like that and combine in huge csv file, then normal processing. Can add distinctions (monthly/yearly) series,	save them with the distinctions, then just load appropriate file. Maybe make all possible distinctions and then write a set combiner method
* Write result in txt file
* Clean Up code and remove unnecessary stuff
* Make theta inclusion nicer

## Question

* why a split ratio for test if that are just our 6/12 values defined in the line therefore exclude these horizons and use them as seperate test?
* 
