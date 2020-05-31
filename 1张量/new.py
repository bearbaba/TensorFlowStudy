import tensorflow as tf


def creat_tensor():
    import tensorflow as tf
    a = tf.constant(3, dtype=tf.int32, shape=(2, 3))
    print(a)


def change_tensor_dtype():
    a = tf.constant(12, dtype=tf.int32, shape=(2, 3))
    a = tf.cast(a, dtype=tf.float32)
    print(a)


if __name__ == '__main__':
    creat_tensor()
    change_tensor_dtype()
