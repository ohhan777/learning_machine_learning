import numpy as np

input_size = 1000
hidden_size = 100
output_size = 10
batch_size = 64

x = np.random.randn(batch_size, input_size)  #input
t = np.random.randn(batch_size, output_size) #label

w1 = np.random.randn(input_size, hidden_size)
w2 = np.random.randn(hidden_size, output_size)

learning_rate = 1e-6

for i in range(500):
    a1 = x.dot(w1)           #a = x*w
    z1 = np.maximum(0, a1)   #relu(a)
    y = z1.dot(w2)

    loss = np.square(y - t).sum()
    print(i, loss)

    #back-propagation
    dy = 2.0 * (y - t)
    dw2 = np.dot(z1.T, dy)
    dz1 = np.dot(dy, w2.T)
    da1 = dz1.copy()
    da1[a1 < 0] = 0
    dw1 = np.dot(x.T, da1)

    w1 -= learning_rate * dw1
    w2 -= learning_rate * dw2




