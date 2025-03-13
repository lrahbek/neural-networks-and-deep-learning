# %%
# Preliminaries
import sys
sys.path.append('../src')
import mnist_loader
import network2
sys.path.append('../fig')
import overfitting
import more_data
import weight_initialization

# %%
# Load data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# %%
# Use the network2 class
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)

# %%
# Bigger hidden layer
net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)

# %%
# Overfitting
overfitting.main('out.txt', 400)

# %%
# Effects of more data
more_data.main()

# %%
# Better weight initialization
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True)

# %%
weight_initialization.main('weight_initialization.txt', 30, 0.1)

# %%
weight_initialization.main('weight_initialization.txt', 100, 0.1)
# %%
