# Kayley Ting, Winter 2020 - Image Recognition Practice
# Objective: Train an MLP Classifier on EMNIST Data to recognize handwritten letters

# EMNIST - Import Data Samples
# Matplotlib - for displaying images
from emnist import extract_training_samples
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import os
from cv2 import cv2
import numpy

# -------------------------------- Import Data + Normalize + Split + Test ---------------------------------------#
# STEP ONE: Import Data, Normalize, Split Train  and Test
# Extract Data from OpenML Website. X matrix of images, y labels
X, y = extract_training_samples('letters')
# Normalize pixel values between 0 and 1
X = X/255.
# Use first 60,000 as training, another 1,000 as testing
X_train, X_test = X[:600], X[600:700]
y_train, y_test = y[:600], y[600:700]
# Number of samples in data set + number of pixels in image
X_train = X_train.reshape(600, 784)
X_test = X_test.reshape(100, 784)

# STEP TWO: Verify that data has been downloaded correctly
img_index = 140
img = X_train[img_index]
print("Image Label: " + str(chr(y_train[img_index]+96)))
plt.imshow(img.reshape((28,28)))
plt.show()

# -------------------------------- MLP Classifier 1, Train + Test + Confusion Matrix  ------------------------------#
# STEP THREE: Create our first MLP Classifier with 1 hidden layer with 50 neurons, run through data 20 times
mlp1 = MLPClassifier(hidden_layer_sizes = (50,), max_iter=20, alpha=1e-4,
                     solver='sgd', verbose=10, tol=1e-4, random_state=1,
                     learning_rate_init=.1)
print("Created first MLP (Multi-Layer Perceptron) Classifier Network")
print("50 neurons, 1 hidden layer, iterate 20 times")
# Use the "fit" command to see how it will perform. Loss should decrease over each iteration
# Loss goes down each time = neuron weights randomly initialized each run
mlp1.fit(X_train, y_train)
print("Training set score: %f" % mlp1.score(X_train, y_train))
print("Test set score: %f" % mlp1.score(X_test, y_test))

# STEP FOUR: Use a Confusion Matrix to visualize the performance and Identify where mistakes are made
# Initialize a list with all the predicted values from the training set
y_pred = mlp1.predict(X_test)
# Confusion Matrix: visualize errors between predictions and actual labels
cm = confusion_matrix(y_test, y_pred)
# This displays a matrix as a plot
plt.matshow(cm)
plt.show()

# STEP FIVE: Let's take a look at the mistaken letter
predicted_letter = 'l'
actual_letter = 'i'
mistake_list = []
for i in range (len(y_test)):
    if(y_test[i] == (ord(actual_letter) - 96) and y_pred[i] == (ord(predicted_letter)-96)):
        mistake_list.append(i)
print("There were " + str(len(mistake_list)) + "times that the letter " +
      actual_letter + " was predicted to be the letter " + predicted_letter + ".")
# We want to see the 3rd mistake made, that's at index  i=4
mistake_to_show = 1
if(len(mistake_list)>mistake_to_show):
    img = X_test[mistake_list[mistake_to_show]]
    plt.imshow(img.reshape((28,28)))
    plt.show()
else:
    print("There weren't that many mistakes! Couldn't show mistake # "+str(mistake_to_show+1) +
          " because there were only "+ str(len(mistake_list)) + " mistakes found!")


# -------------------------------- MLP Classifier 2, Train + Test + Confusion Matrix  ------------------------------#
# STEP SIX: Now let's try another classifier with different parameters
mlp2 = MLPClassifier(hidden_layer_sizes=(100,100,100,100,100,), max_iter=50, alpha=1e-4,
                     solver='sgd', verbose=10, tol=1e-4, random_state=1,
                     learning_rate_init=.1)
mlp2.fit(X_train, y_train)
print("Training set score: %f" % mlp2.score(X_train, y_train))
print("Test set score: %f" %mlp2.score(X_test, y_test))

img = cv2.imread('panda.png')
plt.imshow(img)
plt.show()

# os.walk yields a 3-tuple : 1) Directory Found 2) Sub-directories 3) FileList
for(root,dirs,files) in os.walk('testfolder', topdown=True):
    print (root)
    print (dirs)
    print (files)
    print ('------------------')
