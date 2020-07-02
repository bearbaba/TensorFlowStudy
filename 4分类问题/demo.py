import tensorflow as tf
from tensorflow.keras import datasets

import numpy as np


# f = np.load(r"D:\我的资源\下载位置\mnist.npz")
# x, y = f['x_train'], f['y_train']
# x_test, y_test = f['x_test'], f['y_test']
# f.close()

(x, y), (x_val, y_val) = datasets.mnist.load_data()
x = 2 * tf.convert_to_tensor(x, dtype=tf.float32) / 255. - 1
y = tf.convert_to_tensor(y, dtype=tf.int32)
y = tf.one_hot(y, depth=10)
print(x.shape, y.shape)