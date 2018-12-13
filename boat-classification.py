from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import build_hdf5_image_dataset
import h5py

dataset_train ='train'
dataset_evaluate ='test-boat\primotest.txt'
build_hdf5_image_dataset(dataset_train, image_shape=(128, 38), mode='folder', output_path='dataset.h5', categorical_labels=True, normalize=True)
build_hdf5_image_dataset(dataset_evaluate, image_shape=(128, 38), mode='file', output_path='evaluate.h5', categorical_labels=True, normalize=True)


h5f_t = h5py.File('dataset.h5','r')
X = h5f_t['X']
Y = h5f_t['Y']

h5f_e = h5py.File('evaluate.h5','r')
Xe = h5f_e['X']
Ye = h5f_e['Y']


# Building 'AlexNet'
network = input_data(shape=[None, 38, 128,3])
network = conv_2d(network, 96, 11, activation='relu')
network = max_pool_2d(network, 3)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3)
network = local_response_normalization(network)
network = conv_2d(network, 96, 5, activation='relu')
network = max_pool_2d(network, 3)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='linear')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='linear')
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, loss='categorical_crossentropy', optimizer='momentum',metric='accuracy',learning_rate=0.01)


# Training

model = tflearn.DNN(network)

model.fit(X, Y, validation_set=0.1, n_epoch=30, batch_size=64)
predictions = model.predict(Xe)
print(predictions)

