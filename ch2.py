import numpy as np 

print 'when an image shows up, you will have to dismiss it to continue'

# Import the MNIST data

from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print "Train Shape is ", train_images.shape
print "Length is " , len(train_labels)
print "Labels are ", train_labels
print "Train Shape is ", test_images.shape
print "Length is ", len(test_images)
print "Labels are ", test_labels


#Create a neural network

from keras import models
from keras import layers

myNetwork = models.Sequential()

#Define the Architecture by adding layers
myNetwork.add(layers.Dense(512, activation='relu', input_shape=(28 * 28, )))
myNetwork.add(layers.Dense(10, activation='softmax'))


#Compilation
myNetwork.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics =['accuracy'])

#Display a train image and a test image before reshaping

import matplotlib.pyplot as plt

print 'Display the fifth train image'
digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

print 'Display the fifth test image'
digit = test_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()




#preparing the image data for the network
print 'Reshaping data...'
train_images = train_images.reshape ( (60000, 28 * 28)) # reshapes to 60,000 , 784
train_images = train_images.astype('float32') / 255 # and convert to 0..1 vaues

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255
print 'Data reshaped.'

print 'Categorical...'
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels= to_categorical(test_labels)
print 'Categorical done...'



#importing the data (done via fitting) and training the neural network
print 'Training...'
myNetwork.fit(train_images, train_labels, epochs=5, batch_size=128)
print 'Training complete.'

#Test the model on the test data
print 'Testing the model...'
test_loss, test_acc = myNetwork.evaluate(test_images, test_labels)
print 'test_loss:', test_loss
print 'test_acc:', test_acc



print """ Testing some digits """

imageList = np.random.randint(0, 10000, size=(5))		# put 5 random image indices into an array

for anImageIndex in imageList:
	aTestImage = test_images[anImageIndex]
	print aTestImage.shape
	plt.imshow(aTestImage.reshape( (28, 28)), cmap=plt.cm.binary) # reshape like an image
	plt.show()
	soloImageArray = np.expand_dims(aTestImage, axis =0) # make the (784,) into (1, 784) expected by model
	print soloImageArray.shape
	myNetwork.predict(soloImageArray)				#returns the probabilities for the 10 digits
	print "***************************** ---> The handwritten digit was a ", np.argmax(myNetwork.predict(soloImageArray))




