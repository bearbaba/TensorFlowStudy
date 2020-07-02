import tensorflow as tf

y = tf.constant([0, 1, 2, 3])
y = tf.one_hot(y, depth=4)
print(y)
