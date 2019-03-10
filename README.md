# project

## Current State:

badStuff: A tutorial I tried out at first, but it wasn't good enough to build upon.

torchImpl: A better LSTM for Seq2Seq machine translation but uses torch

kerasImpl: A Goal be a keras implementation of torchImpl


## To do:

* Make sure both implementation are using the same preprocessing -> see cleaning
* Include usage of validation set
* change architecture to the same of torchImplementation 
  * add teacher enforcing
  * add bidirection
  * make sure to use same loss functions for comparability
* Change keras implementation so that it accepts sentences of any length. Right now a memory error appears which needs to be fixed.
