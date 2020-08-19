import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import utils
import tensorflow_addons as tfa

import functools
import numpy as np
import matplotlib.pyplot as plt

from src import dataset_helper as dsh

project_path = '/Users/shivshankar/desktop/deep_learning/alexNet2.0/src'
data_path = '/Users/shivshankar/desktop/deep_learning/alexNet2.0/src/data/'


# Set Global Variables
EPOCHS = 90
VERBOSE = 1
STEPS_PER_EPOCH = 100
BATCH_SIZE = 128
N = 1

# The data set that will be downloaded and stored in system memory
data_set = 'oxford_flowers102'

# Score metrics
acc_scores = []
loss_scores = []
top_5_acc_scores = []

# Setting up top 5 error rate metric
top_5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=5)
top_5_acc.__name__ = 'top_5_acc'


class AlexNet:
    def __init__(self, input_width=227, input_height=227, input_channels=3, num_classes=10, learning_rate=0.01,
                 momentum=0.9, dropout_prob=0.5, weight_decay=.0005):
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.lr = learning_rate
        self.momentum = momentum
        self.dropout_prob = dropout_prob
        self.weight_decay = weight_decay

        def visualize(self, train_data, test_data, info):
            """
            Visualizes the data set giving 9 samples from each of the training and testing data sets and their
            respective labels.
            :param train_data: tf.data.Dataset object containing training data
            :param test_data: tf.data.Dataset object containing the testing data
            :param info: dataset.info for getting information about the dataset (number of classes, samples, etc.)
            :return: n/a
            """
            tfds.show_examples(info, train_data)
            tfds.show_examples(info, test_data)

        def run_training(self, train_data, test_data, val_data, generator=False):
            """
            Build, compile, fit, and evaluate the alexNet using Keras
            :param train_data:
            :param test_data:
            :param val_data:
            :param generator: True is using a generator to train the network
            :return: a trained model object
            """

            model = keras.Sequential()

            # First Layer: Convolutional layer followed by max pooling and batch normalization
            model.add(keras.layers.Conv2D(input_shape=(self.input_width, self.input_height, self.input_channels),
                                          kernel_size=(11, 11),
                                          strides=(4, 4),
                                          padding='valid',
                                          filters=96,       #changed from 3
                                          activation=tf.nn.relu))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.MaxPooling2D(pool_size=(3, 3),       #change to (2, 2)
                                                strides=(2, 2),
                                                padding='valid'))

            # Second Layer: Similar to first layer but biases initialized to 1
            model.add(keras.layers.Conv2D( #if error fill in an input_shape=(x, y, z) argument
                                          kernel_size=(5, 5),
                                          bias_initializer='ones',
                                          strides=(1, 1),
                                          padding='valid',
                                          filters=256,  # changed from 3
                                          activation=tf.nn.relu))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.MaxPooling2D(pool_size=(3, 3),  # change to (2, 2)
                                                strides=(2, 2),
                                                padding='valid'))

            # Third Layer: Just Convolve
            model.add(keras.layers.Conv2D(kernel_size=(3, 3),
                                          strides=(1, 1),
                                          padding='valid',
                                          filters=384,
                                          activation=tf.nn.relu))

            # Fourth Layer: Just Convolve
            model.add(keras.layers.Conv2D(kernel_size=(3, 3),
                                          bias_initializer='ones',
                                          strides=(1, 1),
                                          padding='valid',
                                          filters=384,
                                          activation=tf.nn.relu))

            # Fifth Layer: Convolve and pool
            model.add(keras.layers.Conv2D(kernel_size=(3, 3),
                                          bias_initializer='ones',
                                          strides=(1, 1),
                                          padding='valid',
                                          filters=256,
                                          activation=tf.nn.relu))
            model.add(keras.layers.MaxPooling2D(pool_size=(3, 3),
                                                strides=(2, 2),
                                                padding='valid'))

            # Flatten the output to feed it to the Dense (Fully Connected) Layers
            model.add(keras.layers.Flatten())

            # Sixth Layer: Full Connected
            model.add(keras.layers.Dense(units=4096,
                                         input_shape=(256, 256, 3),     #idk what the input shape here is,
                                                                        # remove the line if it doesnt work??
                                         bias_initializer='ones',
                                         activation=tf.nn.relu))
            model.add(keras.layers.Dropout(rate=self.dropout_prob))

            # Seventh layer: Fully Connected
            model.add(keras.layers.Dense(units=4096,
                                         bias_initializer='ones',
                                         activation=tf.nn.relu))
            model.add(keras.layers.Dropout(rate=self.dropout_prob))

            # Eighth layer: Fully Connected
            model.add(keras.layers.Dense(units=4096,
                                         bias_initializer='ones',
                                         activation=tf.nn.relu))
            model.add(keras.layers.Dropout(rate=self.dropout_prob))

            # Output layer
            model.add(keras.layers.Dense(units=self.num_classes,
                                         activation=tf.nn.softmax))

            model.summary()

            # Compiles the model using SGD optimize and categorical cross entropy loss function. If
            # your data is not one hot encoded, change the loss to 'sparse_categorical_crossentropy' which
            # accepts integer values labels rather than 1-0 arrays.

            #lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=)
            #optimizer = optimizers.SGD(learning_rate=self.lr, momentum=self.momentum, weight_decay=weight_decay)
            optimizer = tfa.optimizers.SGDW(learning_rate=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc', top_5_acc])

            # Fit the model on the training data and validate on the validation data
            if generator:
                file_names  = np.load(data_path + 'file_names.npy')
                num_files = file_names.shape[0]
                del file_names

                model.fit_generator(generator=train_data,
                                    steps_per_epoch=int(num_files // BATCH_SIZE),
                                    epochs=EPOCHS,
                                    verbose=VERBOSE,
                                    validation_data=val_data
                                    )
            else:
                model.fit(train_data,
                          epochs=EPOCHS,
                          validation_data=val_data,
                          verbose=VERBOSE,
                          steps_per_epoch=STEPS_PER_EPOCH)

            # Evaluate the model
            loss, acc, top_5 = model.evaluate(test_data,
                                              verbose=VERBOSE,
                                              steps=5)
            # Append metrics to scores list
            loss_scores.append(loss)
            acc_scores.append(acc)
            top_5_acc_scores.append(top_5)

            return model

        def predictions(self, model, val_images, val_labels, num_examples=1):
            """Displays some examples of the predictions that the newtwork is making on the test data.
            :param model: model object
            :param test_images: tf.data.Dataset object containing training data
            :param test_labels: tf.data.Dataset object containing the testing data
            :return n/a
            """

            preds = model.predict(val_images)

            for i in range(num_examples):
                plt.subplot(1, 2, 1)
                # Plot the first predicted image
                plt.imshow(val_images[i])
                plt.subplot(1, 2, 2)
                # plot bar of confidence of predictions of possible classes for the first image in the test data
                plt.bar([j for j in range(len(preds[i]))], preds[i])
                plt.show()

        def run_experiment(self, n, large_data_set=False, generator=False):
            """

            :param n: number of experiments to perform
            :param large_data_set: True if you want to save the large dataset to hard disk and use generator for training
            :param generator: True if you want to use a generator to train the network
            :return: n/a
            """
            for exp in range(n):
                if large_data_set:
                    dsh.save_data()
                else:
                    train_data, test_data, val_data, info = dsh.load_data()

                    train_data, test_data, val_data, \
                    train_images, train_labels, test_images, test_labels, \
                    val_images, val_labels = dsh.preprocess_data(train_data, test_data, val_data)

                    visualize(train_data, test_data, info)

                    if generator:
                        train_images_file_names = np.load(project_path + 'file_names.npy')
                        train_labels = np.load(project_path + 'oh_labels.npy')
                        train_data_loaded = dsh.DataGenerator(train_images_file_names, train_labels, self.batch_size)
                    else:
                        train_data_loaded = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
                        train_data_loaded = train_data_loaded.repeat().shuffle(1020).batch(self.batch_size)

                    test_data_loaded = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
                    test_data_loaded = test_data.repeat().shuffle(1020).batch(self.batch_size)

                    val_data_loaded = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
                    val_data = val_data.batch(self.batch_size)

                    model = run_training(train_data_loaded, test_data_loaded, val_data_loaded)

                    predictions(model, test_images, test_labels, num_examples=5)

            print(acc_scores)
            print('Mean accuracy={}'.format(np.mean(acc_scores)), 'STD DEV accuracy={}'.format(np.std(acc_scores)))
            print('Min_accuracy={}'.format(np.min(acc_scores)), 'Max_accuracy={}'.format(np.max(acc_scores)))




