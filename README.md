# MNIST-CNN
A cnn for MNIST database for number recognition



This is a implementation of deep neural network for the prediction of MINST dataset.

Libraries used : Keras, Numpy, MatplotLib, Pandas

conv2D layer = 32 features, 5*5 conv kernel , relu activation

maxPool 2D = max pool layer of 2*2

Dropout layer which randomly removes 20% of neurons which is a regularization layer which reduces overfitting 

Flatten layer which converts the matrix to a vector fot input to the FC layers

FC layer of 128 neurons relu activation

Final layer of 10 neurons softmax activation

Compile :-
Optimiser : Adam
Loss : Categorical _ crossentropy

Fit:-
batch_size = 200
epoch = 10
