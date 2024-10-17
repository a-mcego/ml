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
y_train_binary = np.eye(num_classes, dtype=bool)[y_train]

# Winnow algorithm parameters
alpha = 1.01  # Promotion/demotion factor
num_features = x_train.shape[1]
weights = np.ones((num_classes, num_features))  # Initialize weights to 1
threshold = 0.0#float(num_features)

# Evaluate the model
def evaluate_winnow(x_test, y_test):
    result = np.einsum("ba,ca->bc", x_test, weights)
    predictions = np.zeros_like(result, dtype=bool)
    for i in range(x_test.shape[0]):
        predictions[i,np.argmax(result[i])] = True

    #predictions = np.einsum("ba,ca->bc", x_test, weights) >= threshold
    accuracy = np.mean(np.logical_and.reduce(predictions == y_test,1))
    return accuracy

# Training the Winnow algorithm
def train_winnow(x_train, y_train, alpha):
    global threshold
    global weights
    #total_trues = 0
    for i in range(x_train.shape[0]):
        x = x_train[i]
        yvec = y_train[i]

        #prediction = np.dot(weights, x) >= threshold
        #total_trues += (np.sum(prediction)-1)
        
        result = np.dot(weights, x)
        prediction = np.zeros_like(result, dtype=bool)
        prediction[np.argmax(result)] = True
        
        for yi in range(num_classes):
            if yvec[yi] == 1 and not prediction[yi]:
                weights[yi] *= np.where(x == 1, alpha, 1)
                weights[yi] /= np.sum(weights[yi])/num_features
            elif yvec[yi] == 0 and prediction[yi]:
                weights[yi] *= np.where(x == 1, 1/alpha, 1)
                weights[yi] /= np.sum(weights[yi])/num_features
    #threshold += total_trues/x_train.shape[0]*10.0

#plt.ion()  # Turn on interactive mode
#fig, ax = plt.subplots()
#im = ax.imshow(np.reshape(weights[1],(28*2,28)), cmap='viridis', vmin=0, vmax=1)  # Display initial weights

for epoch in range(1000000):
    starttime = time.time()
    train_winnow(x_train, y_train_binary, alpha)
    midtime = time.time()
    accuracy = evaluate_winnow(x_train, y_train_binary)
    endtime = time.time()
    print(f"{epoch} {accuracy:.4f} {alpha} {threshold} {midtime-starttime} {endtime-midtime}")
    #im.set_array(np.reshape(weights[1],(28*2,28)))  # Update the image data
    #plt.draw()  # Redraw the current figure
    #plt.pause(0.1)  # Pause to allow the plot to update

#plt.ioff()  # Turn off interactive mode
#plt.show()  # Keep the plot open after the loop
