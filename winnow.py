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
x_train = np.concatenate([x_train,1-x_train],axis=-1)

num_classes = 10
alpha = 1.0001  # Promotion/demotion factor
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
    
    result = np.einsum("ba,ca->bc", x_train, weights)
    
    for i in range(x_train.shape[0]):
        x = x_train[i]
        yvec = y_train[i]

        #result = np.dot(weights, x)
        result_id = np.argmax(result[i])
        if yvec != result_id:
        #zzz, see if this can be extended to non-binary values and maybe negative values. use the log/exp trick where the weights are actually raised to a power, so we can just add things. so then we just add the x value instead of check x==1 :-)
            weights[yvec]      *= np.where(x == 1, alpha,   1)
            weights[result_id] *= np.where(x == 1, 1/alpha, 1)
            weights[yvec]      /= np.sum(weights[yvec])/num_features
            weights[result_id] /= np.sum(weights[result_id])/num_features

for epoch in range(1000000):
    starttime = time.time()
    train_winnow(x_train, y_train, alpha)
    midtime = time.time()
    accuracy = evaluate_winnow(x_train, y_train)
    endtime = time.time()
    print(f"{epoch} {accuracy:.4f} {alpha} {midtime-starttime} {endtime-midtime}")
