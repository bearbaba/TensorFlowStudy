import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt


def drawing(x1, x2, y):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig = plt.figure(figsize=(8, 6))
    ax3d = Axes3D(fig)

    ax3d.scatter(x1, x2, y, color="b", marker="*")

    ax3d.set_xlabel('面积', color="red", fontsize=16)
    ax3d.set_ylabel('房子数量', color="r", fontsize=16)
    ax3d.set_zlabel("价格", color="r", fontsize=16)
    ax3d.set_zlim3d(30, 160)

    plt.show()


class ForecastModule:
    def __init__(self, x1, x2, y):
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.w = None

    def get_parameter_w(self):
        x0 = tf.ones(len(self.x1))
        x = tf.stack((x0, self.x1, self.x2), axis=1)
        y = tf.reshape(self.y, shape=(-1, 1))
        xt = tf.transpose(x)
        xtx_1 = tf.linalg.inv(tf.matmul(xt, x))
        xtx_1xt = tf.matmul(xtx_1, xt)
        w = tf.matmul(xtx_1xt, y)
        w = tf.reshape(w, shape=(3,))
        self.w = w

    def get_pred_y(self, x1, x2):
        y_pred = self.w[1] * x1 + self.w[2] * x2 + self.w[0]
        return y_pred


if __name__ == '__main__':

    x1 = tf.constant(
        [137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00, 106.69, 138.05, 53.75, 46.91, 68.00, 63.02,
         81.26, 86.21])
    x2 = tf.constant([3, 2, 2, 3, 1, 2, 3, 2, 2, 3, 1, 1, 1, 1, 2, 2], dtype=tf.float32)
    y = tf.constant(
        [145.00, 110.00, 93.00, 116.00, 65.32, 104.00, 118.00, 91.00, 62.00, 133.00, 51.00, 45.00, 78.50, 69.65, 75.69,
         95.30])
    forecast_module = ForecastModule(x1, x2, y)
    forecast_module.get_parameter_w()
    y_pred = forecast_module.get_pred_y(x1, x2)
    drawing(x1, x2, y_pred)
