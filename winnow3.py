import numpy as np
import matplotlib.pyplot as plt
import time
import numpy as np
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Print the shapes of the datasets
print("Training data shape:", x_train.shape)
print("Training labels shape:", y_train.shape)
print("Test data shape:", x_test.shape)
print("Test labels shape:", y_test.shape)

x_train = x_train.reshape(x_train.shape[0], -1)
x_train = x_train / 255.0
#x_train = np.concatenate([x_train,1-x_train],axis=-1)
#x_train = np.concatenate([x_train,-x_train],axis=-1)
#x_train = np.concatenate([x_train,x_train*x_train],axis=-1)
#x_train = -x_train

num_classes = 10
alpha = 0.1  # Promotion/demotion factor
num_features = x_train.shape[1]
weights = np.ones((num_classes, num_features))  # Initialize weights to 1

# Evaluate the model
def evaluate_winnow(x_test, y_test):
    result = np.einsum("ba,ca->bc", x_test, weights)
    result_id = np.argmax(result,axis=-1)
    accuracy = np.mean(result_id == y_test)
    return accuracy

# Training the Winnow algorithm. almost. without threshold - we do the highest value as 1
def train_winnow(x_train, y_train, alpha):
    global weights
    
    for i in range(x_train.shape[0]):
        x = x_train[i]
        yonehot = np.zeros(10)
        yonehot[y_train[i]] = 1.0

        result = np.dot(weights, x)
        result_id = np.argmax(result)
        
        result_onehot = np.zeros(10)
        result_onehot[result_id] = 1.0

        delta = yonehot-result_onehot
        delta = delta[:,None]
        
        weights *= np.power(2.0,alpha*x*delta)

for epoch in range(1000000):
    starttime = time.time()
    train_winnow(x_train, y_train, alpha)
    midtime = time.time()
    accuracy = evaluate_winnow(x_train, y_train)
    endtime = time.time()
    print(f"{epoch} {accuracy:.4f} {alpha} {midtime-starttime} {endtime-midtime} {np.mean(weights)}")
    alpha *= 0.8
    if alpha < 1e-7:
        alpha = 0.1
