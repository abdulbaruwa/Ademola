import numpy
import scipy.special


class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learning_rate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # learning rate
        self.lr = learning_rate

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

        # link weight matrices, wih and who
        self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)


        pass


    def train(self, input_list, target_list):
        # convert input_list into 2d array
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T
        # Calculate the signals going into the hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate the signal going into the final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # To get the error (target - your output)
        output_error = targets - final_outputs

        # Back-propagate
        # We walk backwards in the network adjusting the weights based on influence to each node
        # we use the error =(your wanted output - actual output you got)
        # split the error based on weights influence on the node.
        # Adjustment of the weights = weights * error

        hidden_error = numpy.dot(self.who.T, output_error)

        # update the weights for links between the hidden and output layers
        error_correction = (output_error * final_outputs *(1.0 - final_outputs))
        self.who += self.lr * numpy.dot(error_correction, numpy.transpose(hidden_outputs))
        wih_error_correction = (hidden_error * hidden_outputs*(1.0 - hidden_outputs))
        self.wih += self.lr * numpy.dot(wih_error_correction, numpy.transpose(inputs ))
        pass

    def query(self, input):
        hidden_inputs = numpy.dot(self.wih, input)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_output = self.activation_function(final_inputs)

        return final_output

# Test usage
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.3



# Create neural network instance
n = NeuralNetwork(input_nodes, output_nodes, hidden_nodes, learning_rate)

training_data_file = open('/home/datadrive/PythonDev/NeuralNet/NeuralNetwork/TrainAndTestData/mnist_train_100.csv', 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# Go through all records in the training data set
for record in training_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
    pass

print(n.query())