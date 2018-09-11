
recordHistory = True			# When calling the models, record the history of each for potential plotting
plotHistory = True				# Let's be clear if we want the plotting to happen
newStats = True				# Are we overwriting or appending to the stats file?
lFields = ["Layer Size", "Batch Size", "Epochs", "Test Data Loss", "Test Data Acc", "\n"]
statsFile = "metrics.csv"		
hyperParams = "hyperparams.csv"	# where to read the hyperparameters from


#Load the training data and test data
from keras.datasets import reuters
(train_data, train_labels) , (test_data, test_labels) = reuters.load_data(num_words=10000)

#example decoding
word_index=reuters.get_word_index()
reverse_word_index = dict ( [ (value, key) for (key, value) in word_index.items()])
decoded_newswire = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])
print "example newswire: ", decoded_newswire

#Transforming the data for our network
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
	results = np.zeros(  (len(sequences), dimension))
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1.
	return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# one-hot  or categorical encoding.. we could also use the keras built-in to_categorical
def to_one_hot(labels, dimension=46):			# why are we hardcoding the 46
	results = np.zeros((len(labels), dimension))
	for i, label in enumerate(labels):
		results[i,label] = 1
	return results

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

#set aside validation newswires vs. training newswires for evaluation DURING TRAINING
#at this time I'm choosing to re-use the same validation data for each execution of the model
#in future, we might want to do that on a per-iteration basis
x_val=x_train[:1000]
partial_x_train = x_train[1000:]

#set aside validation labels vs. training labels
y_val=one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]


#Prepare a stats file to be written
#This needs to be outside the loop changing the hyperparameters and recompiling models
if newStats == True:
	f=open(statsFile, "w") 		#create new file
	f.write(",".join(lFields))		#write the header line
else:
	f=open(statsFile, "a")		#append to existing file




from keras import models
from keras import layers
import matplotlib.pyplot as plt


def runModel():
	model = models.Sequential()
	model.add(layers.Dense(64, activation='relu', input_shape=(10000,) ))
	model.add(layers.Dense(iMiddleLayerSize, activation='relu'))
	model.add(layers.Dense(46, activation='softmax'))
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])



	print "Fitting the model while feeding it validation data sliced aside..."
	print "Fitting with hyperparams: ", iMiddleLayerSize, iBatchSize, iEpochs
	if recordHistory == True:
		print ("and recording history for each epoch")
		history = model.fit (partial_x_train, partial_y_train, epochs=iEpochs, batch_size=iBatchSize, validation_data = (x_val, y_val))
	else:
		print ("without recording history, and using full training data set")
		model.fit(x_train, y_train, epochs=iEpochs, batch_size=iBatchSize, validation_data=(x_val, y_val))
		#for both options above, 
		#without the validation data tuple, the process doesn't self-report on validation at each epoch on the python sout
	print "Fitting the model DONE."

	return (model, history)


def displayHistoryGraphs(history):
	print "Plotting history..."
	history_dict = history.history
	#get the 4 y-axis values we will plot
	loss_values= history_dict['loss']
	val_loss_values = history_dict['val_loss']
	acc_values=history_dict['acc']
	val_acc_values=history_dict['val_acc']
	#measure any one of them to get the range of epochs we will plot
	rEpochs = range(1, len(loss_values) + 1)
	print "Plotting Training and Validation Loss... (close plot window to continue)"
	plt.plot(rEpochs, loss_values, 'bo', label='Training loss')
	plt.plot(rEpochs, val_loss_values, 'b', label='Validation loss')
	plt.title('Training and validation Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.show()
	print "Plotting Training and Validation Accuracy... (close plot window to continue)"
	plt.plot(rEpochs, acc_values, 'bo', label='Training Accuracy')
	plt.plot(rEpochs, val_acc_values, 'b', label='Validation Accuracy')
	plt.title('Training and validation Accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.show()
	print"Plotting History... DONE"

def recordModelEval(model):
	print "Evaluating the model using the Keras's test data..."
	test_data_loss, test_data_accuracy = model.evaluate(x_test, one_hot_test_labels)	#using the set of keras set of data, returning an unpacked tuple
	print "Test Data Loss: ", test_data_loss
	print "Test Data Acc: ", test_data_accuracy
	print "Evaluating the model DONE."

	lFields=[str(iMiddleLayerSize), str(iBatchSize), str(iEpochs), str(test_data_loss), str(test_data_accuracy), "\n" ]
	print "Recording the data..."
	print ",".join(lFields)
	f.write(",".join(lFields))
	#print "Recording layer size", iMiddleLayerSize
	#print "batch size : ", iBatchSize
	#print "epochs : ", iEpochs
	#print "Test Data Loss: ", test_data_loss
	#print "Test Data Acc: ", test_data_accuracy


#This loop reads the hyperparameters from csv file
# - - - - - - currently only remembers the last row, which is what it calls the model on.
source = open("hyperparams.csv", "r")
lines = source.readlines()
print range(len(lines))[1:]
for lineNumber in range(len(lines))[1:]:		# skip the header line
	if lines[lineNumber].startswith("#"):
		print "Skipping...", lines[lineNumber]
		continue
	hyperValues = lines[lineNumber].strip('\n').split(',')
	print(lines[lineNumber].split(','))
	iMiddleLayerSize = int(hyperValues[0])		
	iBatchSize=int(hyperValues[1])
	iEpochs=int(hyperValues[2])
	print iMiddleLayerSize, iBatchSize, iEpochs

	thisModel , thisHistory = runModel()
	recordModelEval(thisModel)
	if plotHistory == True:
		displayHistoryGraphs(thisHistory)
source.close()

f.close()

print "Random Baseline..."
import copy
test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array = np.array(test_labels) == np.array(test_labels_copy)
print float(np.sum(hits_array)) / len(test_labels)


print "Demonstrate each predictions is a probability distribution over the 46 topics..."
print "Creating predictions for entire test set..."
predictions = thisModel.predict(x_test)
print "The shape of an example prediction is a vector for all the topics: ", predictions[0].shape
print "All predictions add up to one. Example for one data set: ", np.sum(predictions[0])

print "---------------------------------"

