from tensorflow.keras import layers
import tensorflow as tf

x = tf.random.normal([4, 28 * 28])
fc = layers.Dense(512, activation=tf.nn.relu)
h1 = fc(x)
print(fc.kernel)
print(fc.bias.shape)
print(h1)
