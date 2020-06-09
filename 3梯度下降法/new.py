import tensorflow as tf
import numpy as np


class ForecastModule:
    def __init__(self, x, y, learn_rate, iter, display_step):
        self.x = x
        self.y = y
        self.learn_rate = learn_rate
        self.display_step = display_step
        self.iter = iter

    def get_params(self):
        np.random.seed()
        w = tf.Variable(np.random.randn())
        b = tf.Variable(np.random.randn())
        mse = []
        for i in range(0, self.iter + 1):
            with tf.GradientTape() as tape:
                pred = w * x + b
                loss = 0.5 * tf.reduce_mean(tf.square(y - pred))
            mse.append(loss)

            dl_dw, dl_db = tape.gradient(loss, [w, b])

            w.assign_sub(learn_rate * dl_dw)
            b.assign_sub(learn_rate * dl_db)

            if i % display_step == 0:
                print("i:{},Loss:{},w:{},b{}".format(i, loss, w.numpy(), b.numpy()))


if __name__ == '__main__':
    x = np.array(
        [137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00, 106.69, 138.05, 53.75, 46.91, 68, 63.02, 81.26,
         86.21])
    y = np.array(
        [145.00, 110.00, 93.00, 116.00, 65.32, 104.00, 118.00, 91.00, 62.00, 133.00, 51, 45, 78.50, 69.65, 75.69,
         95.30])
    learn_rate = 0.0001
    iter = 10
    display_step = 1  # 每隔display_step次输出一下损失参数
    forecast_module = ForecastModule(x, y, learn_rate, iter, display_step)
    forecast_module.get_params()
