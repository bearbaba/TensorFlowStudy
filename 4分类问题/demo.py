import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers


import numpy as np

# f = np.load(r"D:\我的资源\下载位置\mnist.npz")
# x, y = f['x_train'], f['y_train']
# x_test, y_test = f['x_test'], f['y_test']
# f.close()

(x, y), (x_val, y_val) = datasets.mnist.load_data()
x = 2 * tf.convert_to_tensor(x, dtype=tf.float32) / 255. - 1
y = tf.convert_to_tensor(y, dtype=tf.int32)
print(x.shape, y.shape)

layers.Dense(256, activation='relu')
model = tf.keras.Sequential([
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)
])

with tf.GradientTape() as tape:
    x = tf.reshape(x, (-1, 28 * 28))
    out = model(x)
    y_onehot = tf.one_hot(y, depth=10)

    # 计算差的平方和
    loss = tf.square(out - y_onehot)
    loss = tf.reduce_sum(loss) / x.shape[0]
    grads = tape.gradient(loss, model.trainable_variables)
    optimizers.Optimizer.apply_gradient(loss, model.trainable_variables)
