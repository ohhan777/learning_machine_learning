import tensorflow as tf
import numpy as np

input_size = 1000
hidden_size = 100
output_size = 10
batch_size = 64

x = tf.placeholder(tf.float32, shape=(None, input_size))
t = tf.placeholder(tf.float32, shape=(None, output_size))

w1 = tf.Variable(tf.random_normal((input_size, hidden_size)))
w2 = tf.Variable(tf.random_normal((hidden_size, output_size)))

learning_rate = 1e-6

a1 = tf.matmul(x, w1)
z1 = tf.maximum(a1, 0)

y = tf.matmul(z1, w2)

loss = tf.reduce_sum((y - t)**2)

dw1, dw2 = tf.gradients(loss, [w1, w2])

new_dw1 = w1.assign(w1 - learning_rate * dw1)
new_dw2 = w2.assign(w2 - learning_rate * dw2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    x_value = np.random.randn(batch_size, input_size)
    t_value = np.random.randn(batch_size, output_size)

    for i in range(500):
        loss_value, _, _ = sess.run([loss, new_dw1, new_dw2], feed_dict={x: x_value, t:t_value})
        print(i, loss_value)
