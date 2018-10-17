#####################################################################################################################
#   #   This is a starter code in Python 3.6 for a 2-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delata12, X12 - weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer)
#
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################
"""
Created on Tue Mar  6 23:21:01 2018

@author: Aashaar Panchalan
         Priyadarshini Vasudevan
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class NeuralNet:
    def __init__(self, train, header = True, h1 = 4, h2 = 2):
        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers

        raw_input = pd.read_csv(train,header=None)
        # TODO: Remember to implement the preprocess method
        train_dataset = self.preprocess(raw_input)
        ncols = len(train_dataset.columns)
        nrows = len(train_dataset.index)
        self.X = train_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        self.y = train_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])
        # Split datasets into training and test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)
        scaler = MinMaxScaler()
        #scaler = StandardScaler()
        # Fit only to the training data
        scaler.fit(self.X_train)
        # Now apply the transformations to the data:
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
        
        
        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X_train
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X_train), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X_train), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))
    #
    # TODO: I have coded the sigmoid activation function, you need to do the same for tanh and ReLu
    #

    def __activation(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            #self.__sigmoid(self, x)
            return self.__sigmoid(x)
        elif activation == "tanh":
            #self.__tanh(self, x)
            return self.__tanh(x)
        elif activation == "ReLu":
            #self.__ReLu(self, x)
            return self.__ReLu(x)

    #
    # TODO: Define the function for tanh, ReLu and their derivatives
    #

    def __activation_derivative(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            #self.__sigmoid_derivative(self, x)
            return self.__sigmoid_derivative(x)
        elif activation == "tanh":
            #self.__tanh_derivative(self, x)
            return self.__tanh_derivative(x)
        elif activation == "ReLu":
            #self.__ReLu_derivative(self, x)
            return self.__ReLu_derivative(x)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __tanh(self,x):
        return np.tanh(x);
    
    def __ReLu(self,x):
        return x * (x > 0)
        


    
    
    # derivative of sigmoid function, indicates confidence about existing weight

    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def __tanh_derivative(self,x):
        return 1. - x * x
    
    def __ReLu_derivative(self,x):
        return 1. * (x > 0)


    #
    # TODO: Write code for pre-processing the dataset, which would include standardization, normalization,
    #   categorical to numerical, etc
    #

    def preprocess(self, X):
        df1=X
        for column in df1:
            if df1[column].dtype == 'object':
                df1[column] = df1[column].astype('category').cat.codes.astype('int64') 
                #df1.to_csv("out1.csv", sep=',')
                #using lable encoding
        df1= X.fillna(X.mean())

        #for column in df1.columns[:-1]:
            #mean = df1[column].mean()
            #std = df1[column].std()
            #df1[column] = (df1[column] - mean) / std 

        return df1

    # Below is the training function

    def train(self, activation, max_iterations = 1000, learning_rate = 0.05):
        for iteration in range(max_iterations):
            out = self.forward_pass(activation)
            error = 0.5 * np.power((out - self.y_train), 2)
            self.backward_pass(out, activation)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input

        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)
        print(self.w23)

    def forward_pass(self, activation):
        # pass our inputs through our neural network
        in1 = np.dot(self.X_train, self.w01 )
        #print("in1 : ",in1)
        self.X12 = self.__activation(in1, activation)
        #print("X12 : ",self.X12)
        #print("w12 : ",self.w12)
        in2 = np.dot(self.X12, self.w12)
        self.X23 = self.__activation(in2, activation)
        in3 = np.dot(self.X23, self.w23)
        out = self.__activation(in3, activation)
        return out



    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)

    # TODO: Implement other activation functions

    def compute_output_delta(self, out, activation="sigmoid"):
        if activation == "sigmoid":
            delta_output = (self.y_train - out) * (self.__sigmoid_derivative(out))
        elif activation == "tanh":
            delta_output = (self.y_train - out) * (self.__tanh_derivative(out))
        elif activation == "ReLu":
            delta_output = (self.y_train - out) * (self.__ReLu_derivative(out))
            
        self.deltaOut = delta_output

    # TODO: Implement other activation functions

    def compute_hidden_layer2_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))
        elif activation == "tanh":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))
        elif activation == "ReLu":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__ReLu_derivative(self.X23))

        self.delta23 = delta_hidden_layer2

    # TODO: Implement other activation functions

    def compute_hidden_layer1_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
        elif activation == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))
        elif activation == "ReLu":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__ReLu_derivative(self.X12))
            
        self.delta12 = delta_hidden_layer1

    # TODO: Implement other activation functions

    def compute_input_layer_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_input_layer = np.multiply(self.__sigmoid_derivative(self.X01), self.delta01.dot(self.w01.T))
        elif activation == "tanh":
            delta_input_layer = np.multiply(self.__tanh_derivative(self.X01), self.delta01.dot(self.w01.T))
        elif activation == "ReLu":
            delta_input_layer = np.multiply(self.__ReLu_derivative(self.X01), self.delta01.dot(self.w01.T))

            self.delta01 = delta_input_layer

    # TODO: Implement the predict function for applying the trained model on the  test dataset.
    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function

    def predict(self,activation="sigmoid", header = False):
        in1 = np.dot(self.X_test, self.w01 )
        #print("in1 : ",in1)
        self.X12 = self.__activation(in1, activation)
        #print("X12 : ",self.X12)
        in2 = np.dot(self.X12, self.w12)
        self.X23 = self.__activation(in2, activation)
        in3 = np.dot(self.X23, self.w23)
        out = self.__activation(in3, activation)
        testError = (1/2) * np.power((out - self.y_test), 2)
        #print("TEST ERROR :")
        print(">>>>> MEAN TEST ERROR : ",str(np.mean(testError)))
        #print("======================================================")
        return 0

def calulate(data_set_path):
    print("************ For Sigmoid Activation Function ************")
    neural_network_sigmoid = NeuralNet(data_set_path)
    neural_network_sigmoid.train(activation="sigmoid")
    testError = neural_network_sigmoid.predict(activation="sigmoid")
    
    print("")
    print("************ For tanh Activation Function ************")
    neural_network_tanh = NeuralNet(data_set_path)
    neural_network_tanh.train(activation="tanh")
    testError = neural_network_tanh.predict(activation="tanh")
    
    print("")
    print("************ For ReLu Activation Function ************")
    neural_network_ReLu = NeuralNet(data_set_path)
    neural_network_ReLu.train(activation="ReLu")
    testError = neural_network_ReLu.predict(activation="ReLu")    

if __name__ == "__main__":
    
    
    car_dataset_path = input("Please enter the dataset path for Car Evaluation dataset :")
    iris_dataset_path = input("Please enter the dataset path for Iris dataset : ")
    adult_dataset_path = input("Please enter the dataset path for Adult Census dataset : ")
    
    print("[1] CAR Dataset :")
    print("")
    #car_dataset_path = "car.csv"
    #car_dataset_path="https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    calulate(car_dataset_path)
    print("======================================================")
    print("")
    
    print("[2] IRIS Dataset : ")
    print("")
    #iris_dataset_path = "iris.csv"
    #iris_dataset_path="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    calulate(iris_dataset_path)
    print("======================================================")
    print("")
    
    print("[3] ADULT Dataset")
    print("")
    #data_set_path = "adult.csv"
    #adult_dataset_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    calulate(adult_dataset_path)
    print("======================================================")
    
    
    
    

