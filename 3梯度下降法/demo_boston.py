import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

boston_housing = tf.keras.datasets.boston_housing
(train_x, train_y), (test_x, test_y) = boston_housing.load_data()

print(train_x.shape, train_y.shape)

print(test_x.shape, test_y.shape)

x_train = train_x[:, 5]
y_train = train_y

x_test = test_x[:, 5]
y_test = test_y

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

learn_rate = 0.04
iter = 2000
display_step = 200

np.random.seed(612)
w = tf.Variable(np.random.randn())
b = tf.Variable(np.random.randn())
print(w.numpy())
print(b.numpy())

mse_train = []
mse_test = []

for i in range(0, iter + 1):
    with tf.GradientTape() as tape:
        pred_train = w * x_train + b
        loss_train = 0.5 * tf.reduce_mean(tf.square(y_train - pred_train))

        pred_test = w * x_test + b
        loss_test = 0.5 * tf.reduce_mean(tf.square(y_test - pred_test))

    mse_train.append(loss_train)
    mse_test.append(loss_test)

    dL_dw, dL_db = tape.gradient(loss_train, [w, b])
    w.assign_sub(learn_rate * dL_dw)
    b.assign_sub(learn_rate * dL_db)

    if i % display_step == 0:
        print("i: {}, Train Loss: {}, Test Loss: {}".format(i, loss_train, loss_test))

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.scatter(x_train, y_train, color="blue", label="data")
plt.plot(x_train, pred_train, color="red", label="model")
plt.legend()

plt.subplot(2,2,2)
plt.plot(mse_train, color="blue", linewidth=3, label="train loss")
plt.plot(mse_test,color="red",linewidth=1.5,label="test loss")
plt.legend()


plt.subplot(2,2,3)
plt.plot(y_train,color="blue", marker="o",label="true_price")
plt.plot(pred_train,color="red", marker=".", label="predict")
plt.legend()

plt.subplot(2,2,4)
plt.plot(y_test, color="blue", marker="o", label="true_price")
plt.plot(pred_test, color="red", marker=".", label="predict")
plt.legend()

plt.show()