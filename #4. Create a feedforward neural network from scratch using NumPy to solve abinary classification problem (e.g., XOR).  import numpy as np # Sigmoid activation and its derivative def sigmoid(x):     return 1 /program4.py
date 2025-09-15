

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)   # here x is already sigmoid(x)


X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])


y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(42)

input_neurons = 2
hidden_neurons = 2
output_neurons = 1

W1 = np.random.uniform(-1, 1, (input_neurons, hidden_neurons))
b1 = np.zeros((1, hidden_neurons))
W2 = np.random.uniform(-1, 1, (hidden_neurons, output_neurons))
b2 = np.zeros((1, output_neurons))


epochs = 10000
lr = 0.1  # learning rate
for epoch in range(epochs):
  
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)   # prediction
    
    error = y - a2  # output error
    d_a2 = error * sigmoid_derivative(a2)
    error_hidden = d_a2.dot(W2.T)
    d_a1 = error_hidden * sigmoid_derivative(a1)
   
    W2 += a1.T.dot(d_a2) * lr
    b2 += np.sum(d_a2, axis=0, keepdims=True) * lr
    W1 += X.T.dot(d_a1) * lr
    b1 += np.sum(d_a1, axis=0, keepdims=True) * lr
  
    if (epoch+1) % 1000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

print("\nFinal predictions after training:")
for i in range(len(X)):
    print(f"Input: {X[i]} -> Predicted: {a2[i][0]:.4f}, Actual: {y[i][0]}")

