import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()
print(x.shape, y.shape)


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)

    return x, y


batch_size = 128
db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(10000).batch(batch_size)

db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db = db.map(preprocess).batch(batch_size)

db_iter = iter(db)
samples = next(db_iter)

print(samples[0].shape, samples[1].shape)

model = Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(10, activation=tf.nn.relu),
])

model.build(input_shape=[None, 28*28])
model.summary()