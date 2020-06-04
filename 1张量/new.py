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
    a = tf.constant([[1, 2], [3, 4]])
    print(tf.shape(a))
    a = tf.expand_dims(a, axis=0)
    print(a)


def sub_tensor_ndim():
    a = tf.constant(1, shape=(1, 2, 3, 1, 1, 4))
    print(a.shape)
    b = tf.squeeze(a)
    c = tf.squeeze(a, [0, 3])
    print("b_shape={},c_shape={}".format(b.shape, c.shape))


def swap_tensor_axis():
    a = tf.constant(1, shape=(1, 2, 3))
    a = tf.transpose(a, perm=[1, 0, 2])
    print(a)


def concat_tensors():
    # 指定在0轴上拼接
    a = tf.constant(1, shape=(1, 2))
    b = tf.constant(2, shape=(1, 2))
    c = tf.concat([a, b], axis=0)
    print(c)

    # 指定在1轴上拼接
    a = tf.constant(1, shape=(1, 2))
    b = tf.constant(2, shape=(1, 2))
    c = tf.concat([a, b], axis=1)
    print(c)


def split_tensor():
    a = tf.constant(1, shape=(2, 4))
    print("在0轴上分割，每一份是2时：\n", tf.split(a, num_or_size_splits=2, axis=0))
    print("在1轴上分割，分割份数之比为1：2：1，", tf.split(a, [1, 2, 1], 1))


def change_tensor_shape():
    a = tf.range(24)
    print(a)
    print(tf.reshape(a, [2, 3, 4]))


def stack_tensor():
    a = tf.constant([1, 2, 3])
    b = tf.constant([4, 5, 6])

    # 在轴为0上进行拼接
    print(tf.stack([a, b], axis=0))

    # 在轴为1上进行拼接
    print(tf.stack([a, b], axis=1))


def unstack_tensor():
    a = tf.reshape(tf.range(6), shape=(2, 3))
    b = tf.unstack(a, axis=0)
    print(b)


def get_tensor_data():
    a = tf.constant([[[1, 2, 3],
                      [4, 5, 6]],
                     [[1, 2, 3],
                      [4, 5, 6]]])
    # print(a[0, 1, 1])
    # print(a[0])
    #
    # print(a[0][0][1:])
    # print(a[0][0:2][0:2])
    # print(a[0, 0:2, 0:2])

    a = tf.range(12, delta=2)
    # print(tf.gather(a, [0, 2, 3]))
    a = tf.reshape(a, (2, 3))
    print(tf.gather_nd(a, [[0, 1], [1, 1]]))


def math_operations():
    # 逐元素相加
    a = tf.constant([1, 2, 3])
    b = tf.constant([4, 5, 6])
    c = tf.add(a, b)
    print(c)

    # 逐元素相减
    d = tf.subtract(b, a)
    print(d)

    # 逐元素相乘
    e = tf.multiply(a, b)
    print(e)

    # 逐元素相除
    f = tf.divide(a, b)
    print(f)

    # 逐元素求余
    g = tf.math.mod(a, b)
    print(g)

    print(a - b)
    print(a * b)
    print(a / b)
    print(a + b)
    print(a % b)
    print(a // b)


def math_operations2():
    a = tf.reshape(tf.range(6), shape=(2, 3))
    print(tf.pow(a, 2))
    print(a ** 2)
    print(tf.square(a))

    print(tf.sqrt(tf.cast(a, dtype=tf.float32)))
    print(tf.cast(a, dtype=tf.float32) ** (1 / 2))

    print(tf.math.log(tf.cast(a, dtype=tf.float32)))
    print(tf.exp(tf.cast(a, dtype=tf.float32)))


def tensor_broadcasting():
    a = tf.constant([1, 2, 3])
    b = tf.reshape(tf.range(12), shape=(4, 3))
    print(a + b)


def matmul_tensor():
    a = tf.reshape(tf.range(6), shape=(2,3))
    b = tf.reshape(tf.range(6), shape=(3,2))
    print(a@b)


if __name__ == '__main__':
    # creat_tensor()
    # change_tensor_dtype()
    # type_to_tensor()
    # get_tensor_property()
    # change_tensor_shape()
    # add_tensor_ndim()
    # sub_tensor_ndim()
    # swap_tensor_axis()
    # concat_tensors()
    # split_tensor()
    # stack_tensor()
    # unstack_tensor()
    # get_tensor_data()
    # math_operations()
    # math_operations2()
    # tensor_broadcasting()
    matmul_tensor()