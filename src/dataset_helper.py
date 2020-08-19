import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from tensorflow import keras
from keras import utils

project_path = '/Users/shivshankar/desktop/deep_learning/alexNet2.0/src/'
data_path = '/Users/shivshankar/desktop/deep_learning/alexNet2.0/src/data/'

data_set = 'oxford_flowers102'


class DataGenerator(utils.Sequence):
    def __init__(self, image_file_names, labels, batch_size):
        self.image_file_names = image_file_names
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image_file_names) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, index):
        batch_image = self.image_file_names[index * self.batch_size : (index + 1) * self.batch_size]
        batch_label = self.labels[index * self.batch_size : (index + 1) * self.batch_size]
        return np.array(batch_image, np.array(batch_label))


def load_data():
    """
    Function for loading and augmenting the training, testing, and validation data.
    :return     images and labels as numpy arras (the labels will be one-hot encoded) as well as an
                info object containing information about the loaded dataset.
    """
    train_data, info = tfds.load(name=data_set,
                                 split=[tfds.core.ReadInstruction('train', from_=k, to=k+10, unit='%')
                                        for k in range(0, 100, 10)], with_info=True)
    val_data = tfds.load(name=data_set,
                         split=[tfds.core.ReadInstruction('train', to=k, unit='%') +
                                tfds.core.ReadInstruction('train', from_=k+10, unit='%')
                                for k in range(0, 100, 10)])

    test_data = tfds.load(name=data_set, split='test')

    assert isinstance(train_data, tf.data.Dataset)
    assert isinstance(test_data, tf.data.Dataset)
    assert isinstance(val_data, tf.data.Dataset)

    print(info)

    return train_data, test_data, val_data, info


def save_data(augment=True):
    """
    If you have <16GB of RAM for the data set and want to use the augmented training dataset (with rotations etc)
    you will need to save to hard disk and use a generator to train your networks.
    This function saves the images (including the augmented ones to hard disk.
    :return:
    """

    file_no = 1

    data = tfds.load(name=data_set, split='train')
    assert isinstance(data, tf.data.Dataset)

    labels = []
    file_names = []

    for example in data:
        image, label = example['image'], example['label']

        # Resize images and add to dataset
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [227, 227])      #change to 256 by 256?
        labels.append(label.numpy())
        np.save(data_path + 'NParray_' + str(file_no), image)
        file_names.append('MParray_' + str(file_no))
        file_no += 1

        if augment:

            # Apply rotation to each image and add a copy to the dataset
            image_rot = tf.image.rot90(image)
            labels.append(label.numpy())
            np.save(data_path + 'NParray_' + str(file_no), image_rot)
            file_names.append('NParray_' + str(file_no))
            file_no += 1

            # Left-right and up-down flip images and add copies to dataset
            image_up_flip = tf.image.flip_up_down(image)
            labels.append(label.numpy())
            np.save(data_path + 'NParray_' + str(file_no), image_up_flip)
            file_names.append('NParray_' + str(file_no))
            file_no += 1

            image_left_flip = tf.image.flip_left_right(image)
            labels.append(label.numpy())
            np.save(data_path + 'NParray_' + str(file_no), image_left_flip)
            file_names.append('NParray_' + str(file_no))
            file_no += 1

            # Apply random saturation change and adda copy to the dataset
            image_sat = tf.image.random_saturation(image, lower=0.2, upper=0.8)
            labels.append(label.numpy())
            np.save(data_path + 'NParray_' + str(file_no), image_sat)
            file_names.append('NParray_' + str(file_no))
            file_no += 1

        # One hot encode labels
        print(len(labels))
        labels = np.array(labels)
        labels = utils.to_categorical(labels)

        # Save labels array to disk
        np.save(project_path + 'oh_labels', labels)

        # Save filenames array to disk
        file_names = np.array(file_names)
        np.save(project_path + 'file_names', file_names)

def preprocess_data(train_data, test_data, val_data):
    """
    Preprocess data by applying resizing, and augment the dataset with rotated and translated versions of
    each image to prevent overfitting.
    :param train_data:
    :param test_data:
    :param val_data:
    :return:
    """

    # Take all the samples in the training set, convert them to float32 and resize.

    training_images = []
    training_labels = []

    for example in train_data.take(-1):
        image, label = example['image'], example['label']

        # Resize images and add to dataset
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [227, 227])      #change to 256, 256?
        training_images.append(image.numpy())
        training_labels.append(label.numpy())
    training_images = np.array(training_images)     #if error omit these 3 lines
    training_labels = utils.to_categorical(np.array(training_labels))


    # Repeat with validation and testing set
    validation_images = []
    validation_labels = []

    for example in val_data.take(-1):
        image, label = example['image'], example['label']

        # Resize images and add to dataset
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [227, 227])  # change to 256, 256?
        validation_images.append(image.numpy())
        validation_labels.append(label.numpy())
    validation_images = np.array(validation_images)
    validation_labels = utils.to_categorical(np.array(validation_labels))

    testing_images = []
    testing_labels = []

    for example in test_data.take(-1):
        image, label = example['image'], example['label']

        # Resize images and add to dataset
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [227, 227])  # change to 256, 256?
        testing_images.append(image.numpy())
        testing_labels.append(label.numpy())
    testing_images = np.array(testing_images)
    testing_labels = utils.to_categorical(np.array(testing_labels))

    return train_data, test_data, val_data, training_images, training_labels, \
           testing_images, testing_labels, validation_images, validation_labels





