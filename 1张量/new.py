import tensorflow as tf


def creat_tensor():
    a = tf.constant(3, dtype=tf.int32, shape=(2, 3))
    print(a)
    b = tf.zeros(2)
    c = tf.ones([2, 3])
    print("b=", b)
    print("c=", c)

    d = tf.fill(dims=[2, 3], value=9)
    print("d=", d)

    e = tf.random.normal([2, 2])
    print("e=", e)

    f = tf.random.uniform(shape=[2, 3], minval=0, maxval=10)
    print("f=", f)


def change_tensor_dtype():
    a = tf.constant(12, dtype=tf.int32, shape=(2, 3))
    a = tf.cast(a, dtype=tf.float32)
    print(a)


def type_to_tensor():
    a = [i for i in range(10)]
    print("a_type=", type(a))
    b = tf.convert_to_tensor(a)
    print(b)


def get_tensor_property():
    a = tf.constant(value=2, shape=(2, 3), dtype=tf.float32)
    print(a.ndim)
    print(a.dtype)
    print(a.shape)

    print(tf.size(a))
    print(tf.shape(a))
    print(tf.rank(a))


def add_tensor_ndim():
    a = tf.constant([2, 3])
    b = tf.constant([1, 2])
    print("a_ndim={},\nb_ndim={}".format(tf.rank(a), tf.rank(b)))
    print(b)
    b = tf.reshape(b, (1, 2))
    b = tf.expand_dims(a, axis=1)
    print(b)


def change_tensor_shape():
    a = tf.range(24)
    print(a)
    print(tf.reshape(a, [2, 3, 4]))


if __name__ == '__main__':
    # creat_tensor()
    # change_tensor_dtype()
    # type_to_tensor()
    # get_tensor_property()
    # change_tensor_shape()
    add_tensor_ndim()
