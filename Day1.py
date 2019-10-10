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
OPTIMIZER = SGD() # SGD optimizer, explained later in this chapter
N_HIDDEN = 128
DROPOUT = 0.3
VALIDATION_SPLIT=0.2 # how much TRAIN is reserved for VALIDATION
# data: shuffled and split between train and test sets

(X_train,y_train), (X_test, y_test)= mnist.load_data()

X_train=X_train.reshape(60000,784)
X_test=X_test.reshape(10000,784)

X_train=X_train.astype('float32')
X_test=X_test.astype('float32')

X_train/=255
X_test=X_test/255

print(X_train.shape)
print(X_test.shape)

Y_train=np_utils.to_categorical(y_train,10)
Y_test=np_utils.to_categorical(y_test,10)

model=Sequential()
model.add(Dense(64, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

history=model.fit(X_train,Y_train, batch_size=BATCH_SIZE,epochs=NB_EPOCH, verbose=VERBOSE,validation_split=0.2)

score=model.evaluate(X_test,Y_test,verbose=1)
print("Test score:",score[0])
print("Test accuracy:",score[1])