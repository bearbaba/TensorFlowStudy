import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class ForecastModule:
    def __init__(self, x_array, y_array):
        self.x_array = x_array
        self.y_array = y_array
        self.w = 0
        self.b = 0
        self.x_text = None
        self.y_text = None

    def get_parameter(self):
        x_mean = tf.reduce_mean(self.x_array)
        y_mean = tf.reduce_mean(self.y_array)
        self.w = tf.reduce_sum((self.x_array - x_mean) * (self.y_array - y_mean)) / tf.reduce_sum(
            tf.square((self.x_array - x_mean)))
        self.b = y_mean - self.w * x_mean

    def test_set(self, x_test):
        self.x_text = x_test
        _ = (self.w * self.x_text + self.b)
        self.y_text = _

    def plot_data(self):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.figure()
        plt.scatter(self.x_array, self.y_array, color="red", label="销售记录")
        plt.scatter(self.x_text, self.y_text, color="blue", label="预测房价")
        plt.plot(self.x_text, self.y_text, color="green", label="拟合直线", linewidth=2)

        plt.xlabel("面积（平方米）", fontsize=14)
        plt.ylabel("价格（万元）", fontsize=14)

        plt.xlim((40, 150))
        plt.xlim((40, 150))

        plt.suptitle("商品房销售价格评估系统", fontsize=20)

        plt.legend(loc="upper left")
        plt.show()


if __name__ == '__main__':
    x = tf.constant(
        [137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00, 106.69, 138.05, 53.75, 46.91, 68.00, 63.02,
         81.26, 86.21])
    y = tf.constant(
        [145.00, 110.00, 93.00, 116.00, 65.32, 104.00, 118.00, 91.00, 62.00, 133.00, 51.00, 45.00, 78.50, 69.65, 75.69,
         95.30])
    forecast_module = ForecastModule(x, y)
    forecast_module.get_parameter()
    x_text = tf.constant(
        [137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00, 106.69, 138.05, 53.75, 46.91, 68.00, 63.02,
         81.26, 86.21])
    forecast_module.test_set(x_text)
    forecast_module.plot_data()
