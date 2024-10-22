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
#alpha = 0.005  # Promotion/demotion factor
#alpha_mult = 0.97 # multiply alpha with this after every epoch
alpha = 0.1
alpha_mult = 0.9
num_features = x_train.shape[1]
weights = np.ones((num_classes, num_features))  # Initialize weights to 1
#weights = np.random.randn(num_classes, num_features) * 0.01  # Initialize weights with small random values
biases = np.zeros((num_classes, 1))  # Initialize biases

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
    
def evaluate_winnow(x_test, y_test):
    result = np.einsum("ba,ca->bc", x_test, weights)
    result_id = np.argmax(result,axis=-1)
    accuracy = np.mean(result_id == y_test)
    return accuracy

def train_winnow(x_train, y_train, alpha):
    global weights
    
    for i in range(x_train.shape[0]):
        x = x_train[i]
        yonehot = np.zeros(10)
        yonehot[y_train[i]] = 1.0

        result = np.dot(weights, x)

        delta = yonehot-softmax(result)
        delta = delta[:,None]
        
        weights *= np.power(2.0,alpha*x*delta)


def softmax_batch(x):
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / e_x.sum(axis=0, keepdims=True)

def evaluate_backprop(x_test, y_test):
    global weights, biases
    logits = np.dot(weights, x_test.T)# + biases
    predictions = np.argmax(softmax_batch(logits), axis=0)
    accuracy = np.mean(predictions == y_test)
    return accuracy

def train_backprop(x_train, y_train, learning_rate):
    global weights, biases
    m = x_train.shape[0]
    
    for i in range(m):
        x = x_train[i].reshape(-1, 1)
        yonehot = np.zeros((num_classes, 1))
        yonehot[y_train[i]] = 1.0

        # Forward pass
        logits = np.dot(weights, x)# + biases
        probs = softmax_batch(logits)

        # Compute gradients
        dL_dlogits = probs - yonehot
        dL_dw = np.dot(dL_dlogits, x.T)
        dL_db = dL_dlogits

        # Update weights and biases
        weights -= learning_rate * dL_dw
        #biases -= learning_rate * dL_db

train = train_winnow
evaluate = evaluate_winnow

#train = train_backprop
#evaluate = evaluate_backprop

for epoch in range(1000000):
    starttime = time.time()
    train(x_train, y_train, alpha)
    midtime = time.time()
    accuracy = evaluate(x_train, y_train)
    endtime = time.time()
    print(f"{epoch} {accuracy:.4f} {alpha} {midtime-starttime} {endtime-midtime} {np.mean(weights)}")
    alpha *= alpha_mult
