# Simple-CNN-for-MNIST
* Author: Volodymyr Lukovych

## Features
This network has an input layer, two convolutional layers and one fully connected output layer. The size of the input layer is [28,28]. Convolutional layers have filters with size [6,6] and stride=2, activation function - ReLU.
The output layer has a softmax activation function. The first convolutional layer has 12 filters and the second - 32 filters. The total number of learned parameters of network is 19430.
The network is implemented in C.

## Learning
The SGD method was used to train the network. The training set was shuffled before each epoch. The initial value of the learning rate is 0.01. During the learning process, the learning rate gradually decreases. 
Training is performed on those elements of the training set that are recognized incorrectly, or if the deviation of the network output from the target value exceeds a certain threshold value. 
This threshold value also decreases during the learning process. With the learning parameters specified in the program, the network achieves an accuracy 0.9952 on the test set.

## Running the Network
The following steps are required.
- Create an empty console app project in Microsoft Visual Studio
- Add to project both source files
- Build the project
- Place a folder with dataset files somewhere on your hard drive
- Create a text file named "mnist_data_path.txt". It must contain the full path to the dataset folder.
- Place the file "mnist_data_path.txt" in the same directory as the executable file of our project.
- Run the executable.

## Dataset
The MNIST dataset can be downloaded from http://yann.lecun.com/exdb/mnist/
