from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from keras.layers import Lambda, Input, Dense, Concatenate
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K, Sequential

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


from keras.models import Model




def create_baseline():
    # create model

    #branch(A)
    latent_dim =2
    a = Input(shape=(28,28))
    b = Dense(10, input_shape=(28, 28), kernel_initializer='normal', activation='relu')(a)

    model_a = Model(inputs=a, outputs=b)

    c = Input(shape=(latent_dim,), name='z_sampling')
    d = Concatenate([model_a.output, c])
    e = Dense(1, kernel_initializer='normal', activation='sigmoid')(d)

    model_a = Model(inputs=[model_a.output, c], outputs=e)


    # Compile model. We use the the logarithmic loss function, and the Adam gradient optimizer.
    model_a.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model_a)

    return model_a

def prepare_model():
    # MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    return([x_train, x_test])


def main():
    prepare_model()
    model_A = create_baseline()
    print(model_A.summary())

if __name__=='__main__':
    main()