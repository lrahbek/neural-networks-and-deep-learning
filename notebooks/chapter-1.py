# %%
# Preliminaries
import sys
sys.path.append('../src')
import mnist_loader
import network
import mnist_average_darkness
import mnist_svm

# %%
# Load data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# %%
# First try
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# %%
# Bigger hidden layer
net = network.Network([784, 100, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# %%
# Small learning rate
net = network.Network([784, 100, 10])
net.SGD(training_data, 30, 10, 0.001, test_data=test_data)

# %%
# Large learning rate
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 100.0, test_data=test_data)

# %%
# Classify by average darkness
mnist_average_darkness.main()

# %%
# Classify using scikit-learn's support vector machine with default settings
mnist_svm.svm_baseline()

