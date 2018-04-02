
# Read and played with the first picture in the file.
input_nodes = 784
hidden_nodes = 100
output_nodes = 10


learning_rate = 0.3


test_date_file = open('/home/datadrive/PythonDev/NeuralNet/NeuralNetwork/TrainAndTestData/mnist_train_100.csv', 'r')
test_date_list = test_date_file.readlines()
test_date_file.close()
print(len(test_date_list))
print(test_date_list[0])

import numpy
import matplotlib.pyplot

all_values = test_date_list[0].split(',')
image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
print(image_array)
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation="None")

# A pixel ranges between 0 and 255 this range is not ideal to pass into the network due to large inter-quartile range
scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
print(scaled_input)
