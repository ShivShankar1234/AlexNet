# AlexNet
Implementation of AlexNet in TensorFlow 2.0

Utilization of Keras and TF 2.0 in the creating of image classifier. The layering architecture and structure of this implementation is described in the paper:

ImageNet Classification with Deep Convolutional Neural Networks:

https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

Hyperparameters can be adjusted in alex_net.py

Datasets were imported from Tensorflow's Dataset Library. Data was preprocesses and training was done using k-fold cross validation.

Highest accuracy achieved on the Oxford Flowers Dataset after 300 epochs was 97.62%
