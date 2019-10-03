from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils
np.random.seed(1671) # for reproducibility
# network and training

NB_EPOCH = 50
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10 # number of outputs = number of digits
OPTIMIZER = Adam(0.1) # SGD optimizer, explained later in this chapter
N_HIDDEN = 128
DROPOUT = 0.3
VALIDATION_SPLIT=0.2 # how much TRAIN is reserved for VALIDATION
# data: shuffled and split between train and test sets