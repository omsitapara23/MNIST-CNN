from keras.datasets import mnist
import matplotlib.pyplot as plt
# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()

#importing the useful libraries
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

'''
Here the pixels are of 1 dim as they are in gray scale now if the image are 
given in RGB than there would be one independent dim for each component 
so in that case the pixels = 3
'''
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#taking out the number of categorical classess
num_classes = y_test.shape[1]

#CREATING THE MODEL

model = Sequential()

#now we add the convolution layer of 32 maps size of 5*5 and relu

model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))

#now we add a max pool layer of 2*2 (for efficency)

model.add(MaxPooling2D(pool_size=(2, 2)))

#this layer is called dropout layer which is for regularization
#this layer randomly drops out 20% of neurons which reduce overfitting

model.add(Dropout(0.2))

#now we flat the matrix to a vector so we can connect it to FC of CNN

model.add(Flatten())

#A FC layer of 128 neurons and relu as an activation function

model.add(Dense(128, activation='relu'))

#final output layer of num_classes neurons each for a seprate category

model.add(Dense(num_classes, activation='softmax'))

#Compiling the model with adam as an optimiser

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fitting the training set with batch size of 200 for 10 epoch
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)


# Final predection
y_pred = model.predict(X_test)

#Evalauation of model
scores = model.evaluate(X_test, y_test, verbose=1)
print("CNN Error: %.2f%%" % (100-scores[1]*100))