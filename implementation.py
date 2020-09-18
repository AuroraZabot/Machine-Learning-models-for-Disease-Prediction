#-------------------------------------------------------------------------------
#Preprocessing
#-------------------------------------------------------------------------------
#Libraries for preprocessing
#From the documentation: the os module provides a portable way of using 
#operating system dependent functionality
import os
#Numpy is the fundamental package for scientific computing 
import numpy as np
#Module for plot images and graphs
import matplotlib.pyplot as plt
#Module for opencv, we would use it to resize the images in the first dataset
import cv2

#Pillow: load, manipulate and save images in many file formats
from PIL import Image
#Useful to display loops with a progress meter
from tqdm import tqdm
#Keras module to categorize outputs
from keras.utils.np_utils import to_categorical
#Keras module used to data augmentation
from keras.preprocessing.image import ImageDataGenerator
#Useful tool to split data into train and test
from sklearn.model_selection import train_test_split

#Function to load images (.png) and transpose them into an array of images
def load(directory):
    #Initialization of the array
    image_array = []
    #Read multiple RGB images 
    read = lambda imname: np.asarray(Image.open(imname).convert('RGB'))
    #For all the images contained in the directory given by the path
    for imgs in tqdm(os.listdir(directory)):
        #This method concatenates various path components with exactly one 
        #directory separator following each non-empty part except the last 
        #path component
        path = os.path.join(directory, imgs)
        #Split the path name into a pair root and extension. We're interested 
        #in the second
        root, figtype = os.path.splitext(path)
        #Since all the images in our dataset are ".png", we would select those 
        #ones
        if figtype == ".png":
            #If the file is a .png, load it
            img = read(path)
            #Resize of images for the first dataset
            img = cv2.resize(img, (210, 135))
            #Add the image into the array of images
            image_array.append(np.array(img))
    #Return the final array of the entire folder
    return image_array

#Load of the train and test set; the train folder is of course bigger than the
#test one:
#Data1
benign_train = np.array(load('data1/train/benign'))
malign_train = np.array(load('data1/train/malignant'))
benign_test = np.array(load('data1/test/benign'))
malign_test = np.array(load('data1/test/malignant'))

#Data2
#benign_train = np.array(load('data2/train/benign'))
#malign_train = np.array(load('data2/train/malignant'))
#benign_test = np.array(load('data2/test/benign'))
#malign_test = np.array(load('data2/test/malignant'))

#We would create the labels: benignant cancer -> 0, malignant -> 1
benign_train_label = np.zeros(len(benign_train))
malign_train_label = np.ones(len(malign_train))
benign_test_label = np.zeros(len(benign_test))
malign_test_label = np.ones(len(malign_test))

#Now that we have the various components, we can merge the data
X_train = np.concatenate((benign_train, malign_train), axis = 0)
Y_train = np.concatenate((benign_train_label, malign_train_label), axis = 0)
X_test = np.concatenate((benign_test, malign_test), axis = 0)
Y_test = np.concatenate((benign_test_label, malign_test_label), axis = 0)

#In order not to have all the elements aligned due to concatenate, we would
#shuffle the data
#We would use np.arange that return evenly spaced values within an interval
X_shape = X_train.shape[0]
train_shuffle = np.arange(X_shape)
#This function only shuffles the array along the first axis of a multi-dimensional array
np.random.shuffle(train_shuffle)
#Train shuffle
X_train = X_train[train_shuffle]
Y_train = Y_train[train_shuffle]

#Same operation on the test set
test_shuffle = np.arange(X_test.shape[0])
np.random.shuffle(test_shuffle)
#Test shuffle
X_test = X_test[test_shuffle]
Y_test = Y_test[test_shuffle]

#Since we have just a binary classification, we would categorize the train and
#test set into two categories (0, 1)
Y_train = to_categorical(Y_train, num_classes= 2)
Y_test = to_categorical(Y_test, num_classes= 2)

#Splitting train and test set in 80/20% as usual appraoch; random state is set 
#to make the execution reproducible
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, 
                                                    test_size = 0.2, 
                                                    random_state = 1)

#Display some images and their classification:
#Data1
width = 30
height = 15

#Data2
#width = 7
#height = 7

#Plot of images with given height and width
fig = plt.figure(figsize = (width, height))
#Number or rows and columns for the plot
cols = 4
rows = 3

#For the total number of plot we want to display (rows*cols)
for i in range(1, cols*rows + 1):
    #Subplot some images
    ax = fig.add_subplot(rows, cols, i)
    #Depending on the labels, we would print "Benign" or "Malignant"
    if np.argmax(Y_train[i]) == 0:
        ax.title.set_text('Benign')
    else:
        ax.title.set_text('Malignant')
    #Show the plot
    plt.imshow(x_train[i], interpolation = 'nearest')
plt.show()

#Data augmentation: done in order to increase the size of training set, we would
#randomly flip images both horizontally (3rd param) and vertically (4th param)  
#in order to obtain representative data but from a different point of view
datagen = ImageDataGenerator(zoom_range = 2, rotation_range = 90,
                             horizontal_flip = True, vertical_flip = True)
#-------------------------------------------------------------------------------
#Implementation
#-------------------------------------------------------------------------------
#Libraries for implementation
#Used in the confusion matrix computation, it's used to create iterators (in our
#case, it is used to substitute a nested for-loop)
import itertools
#Garbage collection module, useful in order to make our computation a little bit
#faster
import gc

#Elements used to construct the model; the first are layers
from keras import layers
#Start the configuration of the model for the training
from keras.models import Sequential
#There are many optimizers in keras, we've chosen Adam since is one of the best
#in this sense
from keras.optimizers import Adam
#DenseNet pretrained weights on ImageNet; in many papers is used in this field
from keras.applications import DenseNet201
#The first callbacks is used to save the best model at a certain epoch, the second
#one to decreasing the learning rate if the model start decreasing the metric
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
#Stuff to analyze how our model works; in particular, the confusion matrix is useful
#tool to analyze missclassifications (that is a very important metric in this 
#particular field), accuracy_score outputs the accuracy metric while 
#classification_report build a text to show the main classification metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

#We would define a function for create our model
def model_architecture():
    #We would use the pretrained model DenseNet, as we've already written before;
    #we set that weights are pretrained on imagenet, then the dimension of images
    #depending on the dataset in use
    #Dataset 1
    height = 135
    width = 210
    channels = 3
    
    #Dataset 2 
    #height = 50
    #width = 50
    #channels = 3

    spine = DenseNet201(weights = 'imagenet', include_top = False, 
                        input_shape = (height, width, channels))
    
    #Starting point for the configuration of the model for training
    model = Sequential()
    #Add pretrained model
    model.add(spine)
    #Global average pooling for spatial data (images, 2D)
    model.add(layers.GlobalAveragePooling2D())
    #Dropout of 50% in order to avoid overfitting
    model.add(layers.Dropout(0.5))
    #Batch Normalization to avoid the vanishing of the gradient
    model.add(layers.BatchNormalization())
    #Two dense layers with a softmax activation function for binary classification
    model.add(layers.Dense(2, activation = 'softmax'))
    
    #Since we've a binary classification, we would use binary crossentropy;
    #We've chosen Adam optimizer since is one of the best optimizers currently
    #Definition of the learning rate; in our case 0.0001
    learning_rate = 1e-4
    model.compile(loss = 'binary_crossentropy', optimizer = 
                  Adam(lr = learning_rate), metrics = ['accuracy'])
    
    #Output of the function
    return model

#Creation of the model
model = model_architecture()
#Brief summary of the model as well
model.summary()

#Callbacks functions are really useful tools used to check internal states or 
#statistics during training
#ReduceLROnPlateau is useful to reduce the learning rate when a metric stops growing
#In our case, we look at the accuracy parameter, we reduce of 0.2 the new learning
#rate, we would wait 5 epochs before decreasing the learning rate if the metrics
#stop improving, we want messages and we set 0.0000001 as the lower threshold
#for the learning rate
reduce_larning_rate = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.2, 
                                        patience = 5, verbose = 1, min_lr = 1e-7)

#ModelCheckpoint save the model after every epoch; in our case, we would save it
#in "filepath", we would monitor the accuracy as well, producing messages, 
#the last best model would not be overwritten and due to the fact we're looking
#at the accuracy, we would save just the maximized accuracy
path = 'checkpoint.hdf5'
checkpointer = ModelCheckpoint(path, monitor = 'val_accuracy', verbose = 1, 
                                save_best_only = True, mode = 'max')

#Batch size has been chosen to be 8/10; the motivation is explained in the report
#The different size depends on the dimension of the training set, since in the 
#fitting of the model we would use this metric also for the "steps_per_epochs"
#and we need an integer
#Dataset 1 
batch = 20
#batch = 15
#batch = 10

#Dataset 2
#batch = 10
#batch = 23

#Parameter to train the model; more insights in the description below
gen = datagen.flow(x_train, y_train, batch_size = batch)
steps = x_train.shape[0] / batch
n_epochs = 10
call = [reduce_larning_rate, checkpointer]

#Training the model on the training set using a batch size defined; the total number
#of steps has been set as suggested in the documentation (looking at the size of
#the training set) and total number of epochs; then we would test our model on
#the test set using the callbacks defined before
training = model.fit_generator(gen, steps_per_epoch = steps, epochs = n_epochs, 
                                validation_data = (x_test, y_test), callbacks = call)
#-------------------------------------------------------------------------------
#Results
#-------------------------------------------------------------------------------
#Load the best achieved model
model.load_weights('checkpoint.hdf5')
#Testing the prediction of the model
y_pred = np.argmax(model.predict(x_test), axis = 1)
y_true = np.argmax(y_test, axis = 1)
#Accuracy classification (normalized)
score = accuracy_score(y_true, y_pred)

#Number of iterations
iterations = 10
#Initialization of the array
predictions = []
#Creates the bar as we've done in the preprocessing part in range of the number
#of iterations we've insert
for i in tqdm(range(iterations)):
    #Method that generates predictions for the input samples; 
    preds = model.predict_generator(datagen.flow(X_test, batch_size = batch, shuffle = False), 
                                    steps = len(X_test)/batch)
    #Add the elements to the array
    predictions.append(preds)
    #Garbage collection of the process (to speed up the computation)
    gc.collect()
    
#Useful tool to build a text to show the main classification metrics; in order
#to fill it, we would build the predicted Ys and the true labels Y_true
Y_pred_tta = np.mean(predictions, axis = 0) 
Y_predicted = np.argmax(Y_pred_tta, axis = 1)
Y_true = np.argmax(Y_test, axis = 1)
report = classification_report(Y_true, Y_predicted)

#Some outputs to evaluate our model; the explanation about what they measure can
#be found in the report
print('\n Accuracy achieved:', score)
print('\n', report)

#Predictions on the input samples, used for the computation of the confusion matrix
Y_pred = np.argmax(model.predict(X_test), axis = 1)

#Function defined to plot the confusion matrix; in input we would have the confusion
#matrix computed through sklearn, the labels and the title of the plot
#I found the tutorial at https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
def plot_confusion_matrix(cm, classes, title, colormap, normalize):
    #Plot of the confusion matrix
    plt.imshow(cm, interpolation = 'nearest', cmap = colormap)
    plt.title(title)
    plt.colorbar()
    
    #We would create a square divided in two by two (total of four subsquares)
    tick_marks = np.arange(len(classes))
    #Plot of the axis of the sub-squares
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    #Construction of the figure, following https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
    fmt = 'd'
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')
    
    #Plot of the confusion matrix
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#Initialization of the parameters for the confusion matrix
#Confusion matrix (offered by sklearn)
confusion_mx = confusion_matrix(Y_true, Y_pred)
#Title of the plot
title = 'Confusion matrix for Skin Cancer'
#Labels (binary classification)
labels = ['Benig, 0', 'Malignant, 1']
#Colormap in use
cmap = plt.cm.Reds
#Normalization
normalization = False

#Plot of the function
plot_confusion_matrix(confusion_mx, labels, title, cmap, normalization)
#Output of the confusion matrix without normalization, in order to be more clear
print('Confusion matrix for Skin Cancer, without normalization: \n', confusion_mx)

#Initialization of arrays containing correct/miss-classifications
correct = []
miss = []
#Dimension of the classification set
length = len(Y_test)
#Number of images we want to plot
number_of_images = 12

#Array of correct classification
i = 0
#For i = 0 to the total number of Y labels
for i in range(length):
    #If correct classify: append
    if(np.argmax(Y_test[i]) == np.argmax(Y_pred_tta[i])):
        correct.append(i)
    #When we reach number_of_images as the array size, we don't need to continue
    #the creation of the array
    if(len(correct) == number_of_images):
        break

#Array of missclassifications
j = 0
#For j = 0 to the total number of Y labels
for j in range(length):
    #If missclassify: append
    if(not np.argmax(Y_test[j]) == np.argmax(Y_pred_tta[j])):
        miss.append(j)
    #When we reach number_of_images as the array size, we don't need to continue
    #the creation of the array
    if(len(miss) == number_of_images):
        break
    
#Finally, we would plot the first 12 images with their labels; the distinction 
#between the two dimensions of images is the same given at the beginning 
#Data1
width = 30
height = 15

#Data2
#width = 12
#height = 12

#Small function to generate labels for the plot
def classification(code):
    #If the label is 0: benign
    if code == 0:
        return 'Benign'
    #If the label is 1: malignant
    else:
        return 'Malignant'

#Plot of images with given height and width
fig = plt.figure(figsize = (width, height))
#Number or rows and columns for the plot
cols = 4
rows = 3

#Plot of the images: first correct classifications
for i in range(len(correct)):
    #Subplot some images
    ax = fig.add_subplot(rows, cols, i + 1)
    #Predicted and actual label
    predicted_corr = np.argmax(Y_pred_tta[correct[i]])
    actual = np.argmax(Y_test[correct[i]])
    #Axis titles, proportinal to the given label
    ax.set_title('Predicted result: ' + classification(predicted_corr) + '\n' + 
                 'Actual result: ' + classification(actual))
    #Show the plot
    plt.imshow(X_test[correct[i]], interpolation = 'nearest')
plt.show()

#Plot of images with given height and width
fig = plt.figure(figsize = (width, height))
#Number or rows and columns for the plot
cols = 4
rows = 3

#Plot of the missclassifications; this is quite a good way to see which images
#causes a missclassification during the process
for j in range(len(miss)):
    #Subplot some images
    ax = fig.add_subplot(rows, cols, j + 1)
    #Predicted and actual label
    predicted_miss = np.argmax(Y_pred_tta[miss[j]])
    actual = np.argmax(Y_test[miss[j]])
    #Axis titles, proportinal to the given label
    ax.set_title('Predicted result: ' + classification(predicted_miss) + '\n' + 
                 'Actual result: ' + classification(actual))
    #Show the plot
    plt.imshow(X_test[miss[j]], interpolation = 'nearest')
plt.show()