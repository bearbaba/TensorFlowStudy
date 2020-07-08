from tensorflow.keras import layers, Sequential
import tensorflow as tf

fc1 = layers.Dense(256, activation=tf.nn.relu)  # 隐藏层1
fc2 = layers.Dense(128, activation=tf.nn.relu)  # 隐藏层2
fc3 = layers.Dense(64, activation=tf.nn.relu)  # 隐藏层3
fc4 = layers.Dense(10, activation=tf.nn.relu)  # 输出层

x = tf.random.normal([4,28*28])
h1 = fc1(x)
h2 = fc2(h1)
h3 = fc3(h2)
h4 = fc4(h3)

model = Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(10, activation=None),
])
out = model(x)
print(out)
