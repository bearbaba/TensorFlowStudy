import tensorflow as tf
a = tf.constant(value=1)
print(a)
b = tf.constant(value=1.0)
print(b)
c = tf.constant(value=1,shape=[2])
print(c)
d = tf.constant(value=2.0,shape=(2,2),dtype='float32')
print(d)

