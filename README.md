# project

## Current State:

badStuff: A tutorial I tried out at first, but it wasn't good enough to build upon.

torchImpl: A better LSTM for Seq2Seq machine translation but uses torch

kerasImpl: Should be a keras implementation of torchImpl


## To do:

* Make sure both implementation are using the same preprocessing -> see cleaning -> done
* Include usage of validation set -> done
* change architecture to the same of torchImplementation -> done
  * make sure to use same loss functions for comparability
* Change keras implementation so that it accepts sentences of any length. Right now a memory error appears which needs to be fixed. (Probably needs to change one hot encoding to word embedding)
* Try to solve the runtime error regarding the allocation failure with cublas (low priority)
* Include custom dataset
* Investigate the maybe better way of Keras Implementation?
